
from .res16unet import Res16UNet34C, Res16UNet34D


def build_backbone(args):
    model = Res16UNet34C(3, 20, args, out_fpn=True)
    return model