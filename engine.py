# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------

import copy
import json
import math
import os
import sys
import time
from typing import Iterable


import numpy as np
import random
import MinkowskiEngine as ME
import wandb
import torch

from utils.seg import mean_iou, mean_iou_scene, cal_click_loss_weights, extend_clicks, get_simulated_clicks
import utils.misc as utils

from evaluation.evaluator_MO import EvaluatorMO

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, train_total_iter: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = 10
    accum_iter = 20

    for i, batched_inputs in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        coords, raw_coords, feats, labels, _, _, click_idx, scene_name, num_obj = batched_inputs
        coords = coords.to(device)
        labels = [l.to(device) for l in labels]
        labels_new = []
        raw_coords = raw_coords.to(device)
        feats = feats.to(device)
        batch_idx = coords[:,0]

        data = ME.SparseTensor(
                            coordinates=coords,
                            features=feats,
                            device=device
                                )

        pcd_features, aux, coordinates, pos_encodings_pcd = model.forward_backbone(data, raw_coordinates=raw_coords)

        #########  1. random sample obj number and obj index #########
        for idx in range(batch_idx.max()+1):
            sample_mask = batch_idx == idx
            sample_labels = labels[idx]
            sample_raw_coords = raw_coords[sample_mask]
            valid_obj_idxs = torch.unique(sample_labels)
            valid_obj_idxs = valid_obj_idxs[valid_obj_idxs!=-1]

            max_num_obj = len(valid_obj_idxs)

            num_obj = np.random.randint(1, min(10, max_num_obj)+1)
            obj_idxs = valid_obj_idxs[torch.randperm(max_num_obj)[:num_obj]]
            sample_labels_new = torch.zeros(sample_labels.shape[0], device=device)

            for i, obj_id in enumerate(obj_idxs):
                obj_mask = sample_labels == obj_id
                sample_labels_new[obj_mask] = i+1

                click_idx[idx][str(i+1)] = []

            click_idx[idx]['0'] = []
            labels_new.append(sample_labels_new)

        click_time_idx = copy.deepcopy(click_idx)
        
        #########  2. pre interactive sampling  #########

        current_num_iter = 0
        num_forward_iters = random.randint(0, 19)

        with torch.no_grad():
            model.eval()
            eval_model = model
            while current_num_iter <= num_forward_iters:
                if current_num_iter == 0:
                    pred = [torch.zeros(l.shape).to(device) for l in labels]
                else:
                    outputs = eval_model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd, 
                                                      click_idx=click_idx, click_time_idx=click_time_idx)
                    pred_logits = outputs['pred_masks']
                    pred = [p.argmax(-1) for p in pred_logits]

                for idx in range(batch_idx.max()+1):
                    sample_mask = batch_idx == idx
                    sample_pred = pred[idx]

                    if current_num_iter != 0:
                        # update prediction with sparse gt
                        for obj_id, cids in click_idx[idx].items():
                            sample_pred[cids] = int(obj_id)

                    sample_labels = labels_new[idx]
                    sample_raw_coords = raw_coords[sample_mask]

                    new_clicks, new_clicks_num, new_click_pos, new_click_time = get_simulated_clicks(sample_pred, sample_labels, sample_raw_coords, current_num_iter, training=True)

                    ### add new clicks ###
                    if new_clicks is not None:
                        click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)

                current_num_iter += 1


        #########  3. real forward pass with loss back propagation  #########

        model.train()
        outputs = model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd, 
                                     click_idx=click_idx, click_time_idx=click_time_idx)

        # loss
        click_weights = cal_click_loss_weights(coords[:,0], raw_coords, torch.cat(labels_new), click_idx)
        loss_dict = criterion(outputs, labels_new, click_weights)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        train_total_iter+=1

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()


        with torch.no_grad():
            pred_logits = outputs['pred_masks']
            pred = [p.argmax(-1) for p in pred_logits]
            metric_logger.update(mIoU=mean_iou(pred, labels_new))

            metric_logger.update(grad_norm=grad_total_norm)
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
 

        if ((i + 1) % 100 == 0):
            wandb.log({
                "train/loss": metric_logger.meters['loss'].avg,
                "train/loss_bce": metric_logger.meters['loss_bce'].avg,
                "train/loss_dice": metric_logger.meters['loss_dice'].avg,

                "train/mIoU": metric_logger.meters['mIoU'].avg,
                "train/total_iter": train_total_iter
                })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, train_total_iter


@torch.no_grad()
def evaluate(model, criterion, data_loader, args, epoch, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    instance_counter = 0
    results_file = os.path.join(args.valResults_dir, 'val_results_epoch_' + str(epoch) + '.csv')
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

            if current_num_clicks != 0:
                click_weights = cal_click_loss_weights(batch_idx, raw_coords, torch.cat(labels), click_idx)
                loss_dict = criterion(outputs, labels, click_weights)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                            for k, v in loss_dict_reduced.items()}

            updated_pred = []

            for idx in range(batch_idx.max()+1):
                sample_mask = batch_idx == idx
                sample_pred = pred[idx]

                sample_mask = sample_mask.to(feats.device)  # Move sample_mask to the same device as feats
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

            if current_num_clicks != 0:
                metric_logger.update(mIoU=mean_iou(updated_pred, labels))
                metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                    **loss_dict_reduced_scaled,
                                    **loss_dict_reduced_unscaled)

            if current_num_clicks == 0:
                new_clicks_num = num_obj[idx]
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num

        instance_counter += len(num_obj)

    f.close()
    evaluator = EvaluatorMO(args.val_list, results_file, [0.5,0.65,0.8,0.85,0.9])
    results_dict = evaluator.eval_results()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update(results_dict)

    return stats
