# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def train_collate_fn_occ(batch):
    imgs, imgs_aug, mask, pids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(imgs_aug, dim=0), torch.stack(mask, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def vis_collate_fn(batch):
    imgs, pids, _, img_path = zip(*batch)
    return torch.stack(imgs, dim=0), pids, img_path

def ranklist_collate_fn(batch):
    imgs, pids, camids, img_path = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_path