# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.modules.attention_block import *
from models.position_embedding import PositionEmbeddingCoordsSine, PositionalEncoding3D, PositionalEncoding1D
from torch.cuda.amp import autocast
from .backbone import build_backbone

import itertools

class Agile3d(nn.Module):
    def __init__(self, backbone, hidden_dim, num_heads, dim_feedforward,
                 shared_decoder, num_decoders, num_bg_queries, dropout, pre_norm,
                 positional_encoding_type, normalize_pos_enc, hlevels,
                 voxel_size, gauss_scale, aux
                 ):
        super().__init__()

        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.hlevels = hlevels
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_bg_queries = num_bg_queries
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.pos_enc_type = positional_encoding_type
        self.aux = aux

        self.backbone = backbone

        self.lin_squeeze_head = conv(
            self.backbone.PLANES[7], self.mask_dim, kernel_size=1, stride=1, bias=True, D=3
        )

        self.bg_query_feat = nn.Embedding(num_bg_queries, hidden_dim)
        self.bg_query_pos = nn.Embedding(num_bg_queries, hidden_dim)


        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=self.mask_dim,
                                                       gauss_scale=self.gauss_scale,
                                                       normalize=self.normalize_pos_enc)
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine",
                                                       d_pos=self.mask_dim,
                                                       normalize=self.normalize_pos_enc)
        else:
            assert False, 'pos enc type not known'

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.masked_transformer_decoder = nn.ModuleList()

        # Click-to-scene attention
        self.c2s_attention = nn.ModuleList()

        # Click-to-click attention
        self.c2c_attention = nn.ModuleList()

        # FFN
        self.ffn_attention = nn.ModuleList()

        # Scene-to-click attention
        self.s2c_attention = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_c2s_attention = nn.ModuleList()
            tmp_s2c_attention = nn.ModuleList()
            tmp_c2c_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()

            for i, hlevel in enumerate(self.hlevels):
                tmp_c2s_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_s2c_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_c2c_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

            self.c2s_attention.append(tmp_c2s_attention)
            self.s2c_attention.append(tmp_s2c_attention)
            self.c2c_attention.append(tmp_c2c_attention)
            self.ffn_attention.append(tmp_ffn_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.time_encode = PositionalEncoding1D(hidden_dim, 200)


    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def forward_backbone(self, x, raw_coordinates=None):
        pcd_features, aux = self.backbone(x)

        with torch.no_grad():
            coordinates = me.SparseTensor(features=raw_coordinates,
                                          coordinate_manager=aux[-1].coordinate_manager,
                                          coordinate_map_key=aux[-1].coordinate_map_key,
                                          device=aux[-1].device)
            coords = [coordinates]
            for _ in reversed(range(len(aux)-1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)

        pcd_features = self.lin_squeeze_head(pcd_features)

        return pcd_features, aux, coordinates, pos_encodings_pcd

    def forward_mask(self, pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=None, click_time_idx=None):

        batch_size = pcd_features.C[:,0].max() + 1

        predictions_mask = [[] for i in range(batch_size)]

        bg_learn_queries = self.bg_query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        bg_learn_query_pos = self.bg_query_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        for b in range(batch_size):

            mins = coordinates.decomposed_features[b].min(dim=0)[0].unsqueeze(0)
            maxs = coordinates.decomposed_features[b].max(dim=0)[0].unsqueeze(0)

            click_idx_sample = click_idx[b]
            click_time_idx_sample = click_time_idx[b]

            bg_click_idx = click_idx_sample['0']

            fg_obj_num = len(click_idx_sample.keys()) - 1

            
            fg_query_num_split = [len(click_idx_sample[str(i)]) for i in range(1, fg_obj_num+1)]
            fg_query_num = sum(fg_query_num_split)

            fg_clicks_coords = torch.vstack([coordinates.decomposed_features[b][click_idx_sample[str(i)], :]
                                       for i in range(1,fg_obj_num+1)]).unsqueeze(0)
            fg_query_pos = self.pos_enc(fg_clicks_coords.float(),
                                     input_range=[mins, maxs]
                                     )

            fg_clicks_time_idx = list(itertools.chain.from_iterable([click_time_idx_sample[str(i)] for i in range(1,fg_obj_num+1)]))
            fg_query_time = self.time_encode[fg_clicks_time_idx].T.unsqueeze(0).to(fg_query_pos.device)
            fg_query_pos = fg_query_pos + fg_query_time

            if len(bg_click_idx)!=0:
                bg_click_coords = coordinates.decomposed_features[b][bg_click_idx].unsqueeze(0)
                bg_query_pos = self.pos_enc(bg_click_coords.float(),
                                        input_range=[mins, maxs]
                                        )  # [num_queries, 128]
                bg_query_time = self.time_encode[click_time_idx_sample['0']].T.unsqueeze(0).to(bg_query_pos.device)
                bg_query_pos = bg_query_pos + bg_query_time

                bg_query_pos = torch.cat([bg_learn_query_pos[b].T.unsqueeze(0), bg_query_pos],dim=-1)
            else:
                bg_query_pos = bg_learn_query_pos[b].T.unsqueeze(0)

            fg_query_pos = fg_query_pos.permute((2, 0, 1))[:,0,:] # [num_queries, 128]
            bg_query_pos = bg_query_pos.permute((2, 0, 1))[:,0,:] # [num_queries, 128]

            bg_query_num = bg_query_pos.shape[0]
            # with torch.no_grad():
            fg_queries = torch.vstack([pcd_features.decomposed_features[b][click_idx_sample[str(i)], :]
                                       for i in range(1,fg_obj_num+1)])

            if len(bg_click_idx)!=0:
                # with torch.no_grad():
                bg_queries = pcd_features.decomposed_features[b][bg_click_idx,:]
                bg_queries = torch.cat([bg_learn_queries[b], bg_queries], dim=0)
            else:
                bg_queries = bg_learn_queries[b]

            src_pcd = pcd_features.decomposed_features[b]

            refine_time = 0

            for decoder_counter in range(self.num_decoders):
                if self.shared_decoder:
                    decoder_counter = 0
                for i, hlevel in enumerate(self.hlevels):

                    pos_enc = pos_encodings_pcd[hlevel][0][b]# [num_points, 128]

                    if refine_time == 0:
                        attn_mask = None

                    output = self.c2s_attention[decoder_counter][i](
                        torch.cat([fg_queries, bg_queries],dim=0), # [num_queries, 128]
                        src_pcd, # [num_points, 128]
                        memory_mask=attn_mask,
                        memory_key_padding_mask=None,
                        pos=pos_enc, # [num_points, 128]
                        query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0) # [num_queries, 128]
                    ) # [num_queries, 128]


                    output = self.c2c_attention[decoder_counter][i](
                        output, # [num_queries, 128]
                        tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0) # [num_queries, 128]
                    ) # [num_queries, 128]

                    # FFN
                    queries = self.ffn_attention[decoder_counter][i](
                        output
                    ) # [num_queries, 128]

                    src_pcd = self.s2c_attention[decoder_counter][i](
                        src_pcd,
                        queries, # [num_queries, 128]
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=torch.cat([fg_query_pos, bg_query_pos], dim=0), # [num_queries, 128]
                        query_pos=pos_enc # [num_points, 128]
                    ) # [num_points, 128]

                    fg_queries, bg_queries = queries.split([fg_query_num, bg_query_num], 0)

                    outputs_mask, attn_mask = self.mask_module(
                                                        fg_queries,
                                                        bg_queries,
                                                        src_pcd,
                                                        ret_attn_mask=True,
                                                        fg_query_num_split=fg_query_num_split)

                    predictions_mask[b].append(outputs_mask)

                    refine_time += 1

        predictions_mask = [list(i) for i in zip(*predictions_mask)]
        
        
        
        out= {
            'pred_masks': predictions_mask[-1],
            'backbone_features': pcd_features
        }

        if self.aux:
            out['aux_outputs'] = self._set_aux_loss(predictions_mask)

        return out


    def mask_module(self, fg_query_feat, bg_query_feat, mask_features, ret_attn_mask=True,
                                fg_query_num_split=None):

        fg_query_feat = self.decoder_norm(fg_query_feat)
        fg_mask_embed = self.mask_embed_head(fg_query_feat)

        fg_prods = mask_features @ fg_mask_embed.T
        fg_prods = fg_prods.split(fg_query_num_split, dim=1)

        fg_masks = []
        for fg_prod in fg_prods:
            fg_masks.append(fg_prod.max(dim=-1, keepdim=True)[0])

        fg_masks = torch.cat(fg_masks, dim=-1)
        
        bg_query_feat = self.decoder_norm(bg_query_feat)
        bg_mask_embed = self.mask_embed_head(bg_query_feat)
        bg_masks = (mask_features @ bg_mask_embed.T).max(dim=-1, keepdim=True)[0]

        output_masks = torch.cat([bg_masks, fg_masks], dim=-1)

        if ret_attn_mask:

            output_labels = output_masks.argmax(1)

            bg_attn_mask = ~(output_labels == 0)
            bg_attn_mask = bg_attn_mask.unsqueeze(0).repeat(bg_query_feat.shape[0], 1)
            bg_attn_mask[torch.where(bg_attn_mask.sum(-1) == bg_attn_mask.shape[-1])] = False

            fg_attn_mask = []
            for fg_obj_id in range(1, fg_masks.shape[-1]+1):
                fg_obj_mask = ~(output_labels == fg_obj_id)
                fg_obj_mask = fg_obj_mask.unsqueeze(0).repeat(fg_query_num_split[fg_obj_id-1], 1)
                fg_obj_mask[torch.where(fg_obj_mask.sum(-1) == fg_obj_mask.shape[-1])] = False
                fg_attn_mask.append(fg_obj_mask)

            fg_attn_mask = torch.cat(fg_attn_mask, dim=0)

            attn_mask = torch.cat([fg_attn_mask, bg_attn_mask], dim=0)

            return output_masks, attn_mask

        return output_masks



    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):

        return [
            {"pred_masks": a} for a in outputs_seg_masks[:-1]
        ]





def build_agile3d(args):

    backbone = build_backbone(args)

    model = Agile3d(
                    backbone=backbone, 
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads, 
                    dim_feedforward=args.dim_feedforward,
                    shared_decoder=args.shared_decoder,
                    num_decoders=args.num_decoders, 
                    num_bg_queries=args.num_bg_queries,
                    dropout=args.dropout, 
                    pre_norm=args.pre_norm, 
                    positional_encoding_type=args.positional_encoding_type,
                    normalize_pos_enc=args.normalize_pos_enc,
                    hlevels=args.hlevels, 
                    voxel_size=args.voxel_size,
                    gauss_scale=args.gauss_scale,
                    aux=args.aux
                    )

    return model