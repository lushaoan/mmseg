#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
__author__ = 'Lu ShaoAn'
__version__ = '1.0'
__date__ = '2021.05.07'
__copyright__ = 'Copyright 2021, PI'


import sys
sys.path.append('../')
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import cv2


config_file = '../configs/unet/pspnet_unet_s5-d16_128x128_40k_chase_db1.py'
checkpoint_file = '../../mmsegmentation_model/unet/pspnet_unet_s5-d16_128x128_40k_chase_db1_20201227_181818-68d4e609.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
img_path = '/media/lsa/MobileDisk3/dataset/CHASEDB1/train/Image_02L.jpg'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
result = inference_segmentor(model, img)
show_result_pyplot(model=model, img=img, result=result)
