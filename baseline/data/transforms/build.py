# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing, RandomOvalErasing


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        if cfg.INPUT.RE_SHAPE == 'rect':
            erase = RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        else:
            erase = RandomOvalErasing(
                probability=cfg.INPUT.RE_PROB,
                shape_mode=cfg.INPUT.RE_SHAPE,
                fill_mode=cfg.INPUT.RE_FILL,
                min_count=cfg.INPUT.RE_MIN_COUNT,
                max_count=cfg.INPUT.RE_MAX_COUNT,
                mean=cfg.INPUT.PIXEL_MEAN,
            )
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            erase,
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
