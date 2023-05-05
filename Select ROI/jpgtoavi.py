from glob import glob
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.signal import find_peaks
import heartpy as hp
from biosppy.signals import tools, ppg
import shutil
import csv

# ################martin################
# addr = 'E:/physio_dataset_verified/only6/'
# subject_name = ['ades6', 'adin6???', 'agus6', 'aice6', 'alana6', 'alex6', 'ali6', 'alina6', 'anggur6', 'apu6', 'aqua6', 'ara6', 'arnold6', 'bunny6', 'cici6', 'citra6', 'dadu6', 'daz6', 'dede6', 'deka6', 'dony6', 'drillyana6', 'eja6', 'fanny6', 'farguso6', 'faye6', 'fitsan6', 'fote6', 'gab6', 'gea6', 'gilberts6', 'gusto6', 'harvey6', 'hello6', 'helly6', 'hexa6', 'ikan6', 'kausar6', 'lea6', 'lili6', 'losory6', 'lyla6', 'mal6', 'murli6', 'nasama6', 'neymar6', 'nick6', 'nusa6', 'ocyra6', 'oren6', 'patrick6', 'peter6', 'phaechan6', 'pulpen6???', 'ran6', 'redoqx6', 'robert6', 'rosali6', 'ryuu6', 'sally6', 'sania6', 'sean6', 'syntax6', 'sysi6', 'toro6', 'tryx6', 'vel6', 'yukibara6', 'yusuf6']
# # print(len(subject_name))

# no=69

# print(no)
# subject_name = subject_name[no]
# print(subject_name)
# DATASET_ROOT = 'E:/physio_dataset_verified/only6/'+subject_name+'/rgb/'

# # the img_dir_path is 'dataset/(subect_name)/rgb'
# img_dir_path = os.path.join(DATASET_ROOT,)
# img_frame_individual_path = glob(os.path.join(img_dir_path, "*.jpg"))
# img_frame_individual_path.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# print(f"Total number of frames: {len(img_frame_individual_path)}")

# # convert all image in the img_Frame_individual_path from 1080 to 720
# for image in img_frame_individual_path:
#     img = cv2.imread(image)
#     img = cv2.resize(img, (1280, 720))
#     cv2.imwrite(image, img)
    
# FPS = 10
# size = (1280, 720)
# save_path = 'm_data/'+subject_name+'.avi'

# out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), FPS, size)

# # read every image and write to video
# for img in img_frame_individual_path:

#     img1 = img.split('_')[2]
#     img2 = img1.split('\\')[1]
#     img_no = int(img2)
#     if img_no%3==0:
#         frame = cv2.imread(img)
#         out.write(frame)


# # release the video
# out.release()
# ################martin################

################our60to10################
addr = 'C:/Users/USER/Desktop/MOST/slidwin_pos/roiroi/our60to10/'
# print(len(subject_name))
'''
0104_shints   0105_cin    0105_hsp    0105_jeff   0105_jihong
0105_long     0105_yee    0105_zizu   0106_brian  0106_chi      0106_sunny
'''
name = '0104_shints'
# the img_dir_path is 'dataset/(subect_name)/rgb'
img_dir_path = os.path.join(addr,)
img_frame_individual_path = glob(os.path.join(img_dir_path, "*.jpg"))
img_frame_individual_path.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
print(f"Total number of frames: {len(img_frame_individual_path)}")

# convert all image in the img_Frame_individual_path from 1080 to 720
for image in img_frame_individual_path:
    img = cv2.imread(image)
    img = cv2.resize(img, (1280, 720))
    cv2.imwrite(image, img)
    
FPS = 10
size = (1280, 720)
save_path = 'our60to10/'+name+'.avi'

out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), FPS, size)

# read every image and write to video
for img in img_frame_individual_path:

    img1 = img.split('.')[0]
    print(img1)
    if img1%6==0:
        frame = cv2.imread(img)
        out.write(frame)


# release the video
# out.release()
################our60to10################
