# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------

import argparse
import torch
from interactive_tool.utils import *
from interactive_tool.interactive_segmentation_user import UserInteractiveSegmentationModel
from interactive_tool.dataloader import InteractiveDataLoader

def main(_):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    dataloader_test = InteractiveDataLoader(config)
    inseg_model_class = UserInteractiveSegmentationModel(device, config, dataloader_test)
    print(f"Using {device}")
    inseg_model_class.run_segmentation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # minimal arguments:
    parser.add_argument('--user_name', type=str, default='user_00')
    parser.add_argument('--pretraining_weights', type=str,
                        default='weights/checkpoint1099.pth')
    parser.add_argument('--dataset_scenes', type=str,
                        default='data/interactive_dataset')
    parser.add_argument('--point_type', type=str, default=None, help="choose between 'mesh' and 'pointcloud'. If not given, the type will be determined automatically")
    
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
    parser.add_argument('--aux', default=True, type=bool, help='whether supervise layer by layer')
    
    parser.add_argument('--device', default='cuda')
    
    
    config = parser.parse_args()

    main(config)
