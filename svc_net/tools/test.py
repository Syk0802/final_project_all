# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir
import time
import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from modeling import build_model_bipa
from utils.logger import setup_logger


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

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

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

    if cfg.DATASETS.NAMES == "market1501":
        flag = "market"
    elif cfg.DATASETS.NAMES == "dukemtmc":
        flag = "duke"
    elif cfg.DATASETS.NAMES == "msmt17":
        flag = "msmt17"
    else:
        raise KeyError(f"Dataset name {cfg.DATASETS.NAMES} is error, please check it!!!")

    idx_list = [i for i in range(51, 80, 3)]


    for i in idx_list:
        ckpt_path = os.path.join(cfg.TEST.WEIGHT, flag, f"resnet50_ibn_a_model_{i}.pth")
        while True:
            time.sleep(5)
            if os.path.exists(ckpt_path):
                _, val_loader, num_query, num_classes = make_data_loader(cfg)
                if "bipa" in ckpt_path:
                    model = build_model_bipa(cfg, num_classes)
                else:
                    model = build_model(cfg, num_classes)

                model.load_param(ckpt_path)
                inference(cfg, model, val_loader, num_query)
                break
            print("------->sleep 5s...")

if __name__ == '__main__':
    main()
