
from .agile3d import build_agile3d

from .criterion import build_mask_criterion

def build_model(args):
    return build_agile3d(args)

def build_criterion(args):
    return build_mask_criterion(args)