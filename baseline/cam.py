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
import torch
from torch.backends import cudnn
from tqdm import tqdm

sys.path.append('.')
from config import cfg
from data import make_data_vis_loader
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


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

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
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_vis_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT, skip_cls=False)
    model = model.cuda()
    model.eval()


    for batch in tqdm(train_loader):
        with torch.no_grad():
            data, pids, img_path = batch
            data = data.cuda()
            cam = model(data, pids, vis=True)
            render(img_path, cam)
        break





if __name__ == '__main__':
    main()
