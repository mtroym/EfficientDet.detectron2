import logging
import torch
from detectron2.modeling import build_anchor_generator
from detectron2.utils.logger import setup_logger
from torch import nn

from .backbone.bifpn import SeparableConvBlock
from .backbone.efficient_utils import MemoryEfficientSwish, Swish

setup_logger(name=__name__)


class Regressor(nn.Module):
    """
    modified by TroyMao
    """

    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        # logging.getLogger(__name__).info(inputs.shape)
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                if len(feat.shape) == 3:
                    feat = feat.unsqueeze(0)
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            # feat = feat.permute(0, 2, 3, 1)
            # feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=0)

        return feats


class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for _ in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for _ in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                if len(feat.shape) == 3:
                    feat = feat.unsqueeze(0)
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            # feat = feat.permute(0, 2, 3, 1)
            # feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
            #                               self.num_classes)
            feats.append(feat)

        feats = torch.cat(feats, dim=0)
        feats = feats.sigmoid()

        return feats


BOXCLS_LAYERS_BY_COEF = [3, 3, 3, 4, 4, 4, 5, 5]


class EfficientDetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape):
        super(EfficientDetHead, self).__init__()
        # fmt: off
        scale = int(cfg.MODEL.EFFICIENTDET.SCALE[-1])
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.EFFICIENTDET.NUM_CLASSES
        num_convs = BOXCLS_LAYERS_BY_COEF[scale]
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
                len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        # cls_subnet = []
        # bbox_subnet = []
        # for _ in range(num_convs):
        #     cls_subnet.append(
        #         nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        #     )
        #     cls_subnet.append(nn.ReLU())
        #     bbox_subnet.append(
        #         nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        #     )
        #     bbox_subnet.append(nn.ReLU())
        #
        # self.cls_subnet = nn.Sequential(*cls_subnet)
        # self.bbox_subnet = nn.Sequential(*bbox_subnet)
        # self.cls_score = nn.Conv2d(
        #     in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        # )
        # self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.bbox_subnet = Regressor(in_channels, num_anchors, num_layers=num_convs,
                                     onnx_export=False)
        self.cls_subnet = Classifier(in_channels, num_anchors, num_classes=num_classes,
                                     num_layers=num_convs, onnx_export=False)

        # Initialization
        # for modules in [self.cls_subnet, self.bbox_subnet,
        #                 # self.cls_score, self.bbox_pred
        #                 ]:
        #     for layer in modules.modules():
        #         if isinstance(layer, nn.Conv2d):
        #             torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
        #             torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # torch.nn.init.constant_(self.cls_subnet.header.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_subnet(feature))
            bbox_reg.append(self.bbox_subnet(feature))
        return logits, bbox_reg
