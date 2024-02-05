import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from torch import Tensor

class SetCriterion(nn.Module):

    def __init__(self, weight_dict, losses):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses


    def multiclass_dice_loss(self, input: Tensor, target: Tensor, eps: float = 1e-6,
                         check_target_validity: bool = True,
                         ignore_mask: Optional[Tensor] = None) -> Tensor:
        """
        Computes DICE loss for multi-class predictions. API inputs are identical to torch.nn.functional.cross_entropy()
        :param input: tensor of shape [N, C, *] with unscaled logits
        :param target: tensor of shape [N, *]
        :param eps:
        :param check_target_validity: checks if the values in the target are valid
        :param ignore_mask: optional tensor of shape [N, *]
        :return: tensor
        """
        assert input.ndim >= 2
        input = input.softmax(1)
        num_classes = input.size(1)

        if check_target_validity:
            class_ids = target.unique()
            assert not torch.any(torch.logical_or(class_ids < 0, class_ids >= num_classes)), \
                f"Number of classes = {num_classes}, but target has the following class IDs: {class_ids.tolist()}"

        target = torch.stack([target == cls_id for cls_id in range(0, num_classes)], 1).to(dtype=input.dtype)  # [N, C, *]


        if ignore_mask is not None:
            ignore_mask = ignore_mask.unsqueeze(1)
            expand_dims = [-1, input.size(1)] + ([-1] * (ignore_mask.ndim - 2))
            ignore_mask = ignore_mask.expand(*expand_dims)

        return self.dice_loss(input, target, eps=eps, ignore_mask=ignore_mask)

    def dice_loss(self, input: Tensor, target: Tensor, ignore_mask: Optional[Tensor] = None, eps: Optional[float] = 1e-6):
        """
        Computes the DICE or soft IoU loss.
        :param input: tensor of shape [N, *]
        :param target: tensor with shape identical to input
        :param ignore_mask: tensor of same shape as input. non-zero values in this mask will be
        :param eps
        excluded from the loss calculation.
        :return: tensor
        """
        assert input.shape == target.shape, f"Shape mismatch between input ({input.shape}) and target ({target.shape})"
        assert input.dtype == target.dtype

        if torch.is_tensor(ignore_mask):
            assert ignore_mask.dtype == torch.bool
            assert input.shape == ignore_mask.shape, f"Shape mismatch between input ({input.shape}) and " \
                f"ignore mask ({ignore_mask.shape})"
            input = torch.where(ignore_mask, torch.zeros_like(input), input)
            target = torch.where(ignore_mask, torch.zeros_like(target), target)

        input = input.flatten(1)
        target = target.detach().flatten(1)

        numerator = 2.0 * (input * target).mean(1)
        denominator = (input + target).mean(1)

        soft_iou = (numerator + eps) / (denominator + eps)

        return torch.where(numerator > eps, 1. - soft_iou, soft_iou * 0.)


    def loss_bce(self, outputs, targets, weights=None):

        pred_masks = outputs['pred_masks']

        loss = 0.0

        for i in range(len(pred_masks)):
            loss_sample = (F.cross_entropy(pred_masks[i], targets[i].long(), reduction="none") * weights[i]).mean()
            loss += loss_sample

        loss = loss/len(pred_masks)

        return {
            "loss_bce": loss
        }

    def loss_dice(self, outputs, targets, weights=None):
        
        pred_masks = outputs['pred_masks']
        loss = 0.0
        for i in range(len(pred_masks)):
            loss_sample = (self.multiclass_dice_loss(pred_masks[i], targets[i].long()) * weights[i]).mean()
            loss += loss_sample
            
        loss = loss/len(pred_masks)
        return {
            "loss_dice": loss
        }

    def get_loss(self, loss, outputs, targets, weights=None):
        loss_map = {
            'bce': self.loss_bce,
            'dice': self.loss_dice
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, weights)

    def forward(self, outputs, targets, weights=None):

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, weights))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, weights)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_mask_criterion(args):

    weight_dict = {
                    'loss_bce': args.bce_loss_coef,
                    'loss_dice': args.dice_loss_coef,
                    }

    losses = args.losses

    if args.aux:
        aux_weight_dict = {}
        for i in range(args.num_decoders*len(args.hlevels)):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(weight_dict, losses)

    return criterion



