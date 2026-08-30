# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import cv2
import numpy as np
import scipy.io
import torch
from torch.backends import cudnn
from tqdm import tqdm

sys.path.append(".")
from config import cfg
from data import make_data_ranklist_loader
from modeling import build_model
from utils.logger import setup_logger


def render(img_path, cams, save_path="cam_img"):
    alpha = 0.6
    beta = 0.4
    gamma = 0
    bs = cams.shape[0]
    os.makedirs(save_path, exist_ok=True)
    for i in range(bs):
        cam = cams[i].squeeze()
        img = cv2.imread(img_path[i])
        img_name = img_path[i].split("/")[-1]
        img_save_path = os.path.join(save_path, img_name)
        cam = cam.data.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = (cam * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        output = cv2.addWeighted(img, alpha, heatmap, beta, gamma)
        cv2.imwrite(img_save_path, output)


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 2048).zero_()  # 1536
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())

            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    # if output_dir and not os.path.exists(output_dir):

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    _, val_loader, num_query, num_classes = make_data_ranklist_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT, skip_cls=False)
    model = model.cuda()
    model.eval()
    featslist = []
    labellist = []
    camidlist = []
    img_pathlist = []

    for batch in tqdm(val_loader):
        with torch.no_grad():
            data, pids, camids, img_path = batch
            data = data.cuda()
            feat = model(data, pids)

            featslist.append(feat)
            labellist += list(pids)
            camidlist += list(camids)
            img_pathlist += list(img_path)

    feats = torch.cat(featslist, dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    feats = feats.data.cpu()

    query_label = labellist[:num_query]
    query_cam = camidlist[:num_query]
    query_feature = feats[:num_query]
    query_img_paths = img_pathlist[:num_query]

    gallery_label = labellist[num_query:]
    gallery_cam = camidlist[num_query:]
    gallery_feature = feats[num_query:]
    gallery_img_paths = img_pathlist[num_query:]

    result = {
        "gallery_f": gallery_feature.numpy(),
        "gallery_label": gallery_label,
        "gallery_cam": gallery_cam,
        "gallery_img_paths": gallery_img_paths,
        "query_f": query_feature.numpy(),
        "query_label": query_label,
        "query_cam": query_cam,
        "query_img_paths": query_img_paths,
    }

    scipy.io.savemat("pytorch_result.mat", result)


if __name__ == "__main__":
    main()
