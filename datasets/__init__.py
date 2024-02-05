from .InterMultiObj3DSegDataset import build as build_multi_obj_dataset
from .InterSingleObj3DSegDataset import build as build_single_obj_dataset

def build_dataset(split, args):
    if args.dataset_mode == 'multi_obj':
        return build_multi_obj_dataset(split, args)
    elif args.dataset_mode == 'single_obj':
        return build_single_obj_dataset(split, args)

    raise ValueError(f'dataset mode {args.dataset_mode} not supported')