# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------

import argparse
import copy
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model, build_criterion
import MinkowskiEngine as ME
from utils.seg import mean_iou_scene, extend_clicks, get_simulated_clicks
import utils.misc as utils

from evaluation.evaluator_MO import EvaluatorMO
import wandb
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)

    # dataset
    parser.add_argument('--dataset_mode', default='multi_obj')
    parser.add_argument('--scan_folder', default='/cluster/work/igp_psr/yuayue/thesis/backup/reproduce/Inter3D/data/preprocess_correct', type=str)
    parser.add_argument('--val_list', default='/cluster/work/igp_psr/yuayue/thesis/backup/reproduce/Inter3D/data/val_scannet_randomEachScene_max10_close.json', type=str)
    parser.add_argument('--train_list', default='', type=str)
    
    # model
    ### 1. backbone
    parser.add_argument('--dialations', default=[ 1, 1, 1, 1 ], type=list)
    parser.add_argument('--conv1_kernel_size', default=5, type=int)
    parser.add_argument('--bn_momentum', default=0.02, type=int)
    parser.add_argument('--voxel_size', default=0.05, type=float)

    ### 2. transformer
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_decoders', default=3, type=int)
    parser.add_argument('--num_bg_queries', default=10, type=int, help='number of learnable background queries')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--pre_norm', default=False, type=bool)
    parser.add_argument('--normalize_pos_enc', default=True, type=bool)
    parser.add_argument('--positional_encoding_type', default="fourier", type=str)
    parser.add_argument('--gauss_scale', default=1.0, type=float, help='gauss scale for positional encoding')
    parser.add_argument('--hlevels', default=[4], type=list)
    parser.add_argument('--shared_decoder', default=False, type=bool)
    parser.add_argument('--aux', default=True, type=bool)

    # evaluation
    parser.add_argument('--val_batch_size', default=1, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint1099.pth', help='resume from checkpoint')

    parser.add_argument('--max_num_clicks', default=20, help='maximum number of clicks per object on average', type=int)

    return parser



def Evaluate(model, data_loader, args, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    instance_counter = 0
    results_file = os.path.join(args.output_dir, 'val_results_multi.csv')
    f = open(results_file, 'w')

    for batched_inputs in metric_logger.log_every(data_loader, 10, header):

        coords, raw_coords, feats, labels, labels_full, inverse_map, click_idx, scene_name, num_obj = batched_inputs
        coords = coords.to(device)
        raw_coords = raw_coords.to(device)
        labels = [l.to(device) for l in labels]
        labels_full = [l.to(device) for l in labels_full]

        data = ME.SparseTensor(
                                coordinates=coords,
                                features=feats,
                                device=device
                                )

        ###### interactive evaluation ######
        batch_idx = coords[:,0]
        batch_size = batch_idx.max()+1

        # click ids set null
        for click_idx_sample in click_idx:
            for obj_id, _ in click_idx_sample.items():
                click_idx_sample[obj_id] = []

        click_time_idx = copy.deepcopy(click_idx)

        current_num_clicks = 0

        # pre-compute backbone features only once
        pcd_features, aux, coordinates, pos_encodings_pcd = model.forward_backbone(data, raw_coordinates=raw_coords)

        max_num_clicks = num_obj[0] * args.max_num_clicks

        while current_num_clicks <= max_num_clicks:
            if current_num_clicks == 0:
                pred = [torch.zeros(l.shape).to(device) for l in labels]
            else:

                outputs = model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd,
                                             click_idx=click_idx, click_time_idx=click_time_idx)
                pred_logits = outputs['pred_masks']
                pred = [p.argmax(-1) for p in pred_logits]


            updated_pred = []

            for idx in range(batch_idx.max()+1):
                sample_mask = batch_idx == idx
                sample_pred = pred[idx]

                sample_feats = feats[sample_mask]

                if current_num_clicks != 0:
                    # update prediction with sparse gt
                    for obj_id, cids in click_idx[idx].items():
                        sample_pred[cids] = int(obj_id)
                    updated_pred.append(sample_pred)

                sample_labels = labels[idx]
                sample_raw_coords = raw_coords[sample_mask]
                sample_pred_full = sample_pred[inverse_map[idx]]

                sample_labels_full = labels_full[idx]
                sample_iou, _ = mean_iou_scene(sample_pred_full, sample_labels_full)

                line = str(instance_counter+idx) + ' ' + scene_name[idx].replace('scene','') + ' ' + str(num_obj[idx]) + ' ' + str(current_num_clicks/num_obj[idx]) + ' ' + str(
                sample_iou.cpu().numpy()) + '\n'
                f.write(line)

                print(scene_name[idx], 'Object: ', num_obj[idx], 'num clicks: ', current_num_clicks/num_obj[idx], 'IOU: ', sample_iou.item())
    
                new_clicks, new_clicks_num, new_click_pos, new_click_time = get_simulated_clicks(sample_pred, sample_labels, sample_raw_coords, current_num_clicks, training=False)

                ### add new clicks ###
                if new_clicks is not None:
                    click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)


            if current_num_clicks == 0:
                new_clicks_num = num_obj[idx]
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num

        instance_counter += len(num_obj)

    f.close()
    evaluator = EvaluatorMO(args.val_list, results_file, [0.5,0.65,0.8,0.85,0.9])
    results_dict = evaluator.eval_results()

def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)
    model.to(device)

    # build dataset and dataloader
    dataset_val, collation_fn_val = build_dataset(split='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.val_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collation_fn_val, num_workers=args.num_workers,
                                 pin_memory=True)

    output_dir = Path(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
      
    Evaluate(model, data_loader_val, args, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script on interactive multi-object segmentation ', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)