from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F

from .mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3

from .utils import _ovewrite_value_param, _VOC_CATEGORIES, IntermediateLayerGetter, Weights, WeightsEnum, _log_api_usage_once, SemanticSegmentation
from loss import KDLoss

__all__ = ["LRASPP", "LRASPP_MobileNet_V3_Large_Weights", "lraspp_mobilenet_v3_large"]


class LRASPP(torch.nn.Module):

    def __init__(
        self, backbone: torch.nn.Module, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = 128,
        criterion=None, criterion_aux=None, args=None) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.kd_loss = KDLoss(temp_factor=args.kd_temp, channel_wise=True).cuda()
        self.args = args
    
    def forward(self, x, gts=None, aux_gts=None, t_out=None, img_gt=None, visualize=None):

        features = self.backbone(x)
        out = self.classifier(features)
        main_out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.training:
            loss1 = self.criterion(main_out, gts)

            if self.args.kd and t_out is not None:
                kd_loss = self.kd_loss(main_out, t_out)
            else:
                kd_loss = torch.FloatTensor([0]).cuda()

            return_loss = [loss1, torch.FloatTensor([0]).cuda()]
            if self.args.kd and t_out is not None:
                return_loss.append(kd_loss)

            return return_loss
        else:
            return main_out
    


class LRASPPHead(torch.nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = torch.nn.Sequential(
            torch.nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            torch.nn.BatchNorm2d(inter_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.scale = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            torch.nn.Sigmoid(),
        )
        self.low_classifier = torch.nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = torch.nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


def _lraspp_mobilenetv3(backbone: MobileNetV3, num_classes: int, **kwargs) -> LRASPP:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels
    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})

    return LRASPP(backbone=backbone, low_channels=low_channels, high_channels=high_channels, num_classes=num_classes, **kwargs)


class LRASPP_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            "num_params": 3221538,
            "categories": _VOC_CATEGORIES,
            "min_size": (1, 1),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#lraspp_mobilenet_v3_large",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 57.9,
                    "pixel_acc": 91.2,
                }
            },
            "_ops": 2.086,
            "_file_size": 12.49,
            "_docs": """
                These weights were trained on a subset of COCO, using only the 20 categories that are present in the
                Pascal VOC dataset.
            """,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


def lraspp_mobilenet_v3_large(
    *,
    weights: Optional[LRASPP_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> LRASPP:
    
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    weights = LRASPP_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _lraspp_mobilenetv3(backbone=backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model