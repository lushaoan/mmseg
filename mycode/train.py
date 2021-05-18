#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.05.12'
__copyright__ = 'Copyright 2021, PI'


import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from mmcv import Config
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
import shutil


if __name__ == '__main__':
    cfg_file = '../configs/pspnet/pspnet_r18-d8_512x1024_20k_ade2016_lsa.py'
    cfg = Config.fromfile(cfg_file)
    cfg.work_dir = '../work_dirs/pspnet_test'
    cfg.gpu_ids = range(1)
    cfg.seed = None

    model = build_segmentor(cfg.model,
                            train_cfg=cfg.get('train_cfg'),
                            test_cfg=cfg.get('test_cfg'))
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=False)
    shutil.copy(cfg_file, os.path.join(cfg.work_dir, cfg_file.split('/')[-1]))