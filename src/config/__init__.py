from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg


def add_efficientdet_config(cfg):
    _C = cfg

    _C.MODEL.EFFICIENTDET = CN()
    _C.MODEL.EFFICIENTDET.NUM_CLASSES = 13
    _C.MODEL.EFFICIENTDET.SCALE = "d3"

    _C.MODEL.EFFICIENT = CN()

    _C.MODEL.EFFICIENT.SCALE = "b3"

    _C.MODEL.EFFICIENT.OUT_FEATURES = ["p{}".format(i) for i in "345"]

    # BIFPN configs.

    _C.MODEL.BIFPN = CN()
    _C.MODEL.BIFPN.IN_FEATURES = ["p{}".format(i) for i in "345"]
    _C.MODEL.BIFPN.SCALE = "d3"

    return _C


def get_efficient_config():
    cfg = get_cfg()
    cfg = add_efficientdet_config(cfg)
    return cfg
