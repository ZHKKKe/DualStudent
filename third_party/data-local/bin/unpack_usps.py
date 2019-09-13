# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import os
import pickle
import sys
import h5py
import math

from tqdm import tqdm
from torchvision.datasets import MNIST
import matplotlib.image
import numpy as np

from PIL import Image
# im = Image.fromarray(A)
# im.save("your_file.jpeg")


def USPS(work_dir, train=True):
    path = os.path.join(work_dir, 'usps.h5')

    subset = 'train' if train else 'test'
    with h5py.File(path, 'r') as hf:
        dataset = hf.get(subset)
        x_data = dataset.get('data')[:]
        y_data = dataset.get('target')[:]
    
    load_set = []
    for x, y in zip(x_data, y_data):
        size = int(math.sqrt(x.shape[0]))
        load_set.append((x.reshape((size, size)), y))
    return load_set

def unpack(source_data, target_dir, start_idx):
    for idx, (image_data, label_idx) in tqdm(enumerate(source_data), total=len(source_data)):
        subdir = os.path.join(target_dir, str(label_idx))
        name = '{}_{}.png'.format(start_idx + idx, str(label_idx))
        os.makedirs(subdir, exist_ok=True)
        im = Image.fromarray(image_data * 255)
        im = im.convert('L').resize((28, 28), resample=Image.BICUBIC)
        im.save(os.path.join(subdir, name))
        # matplotlib.image.imsave(os.path.join(subdir, name), image_data * 255)
    return len(source_data)


work_dir = os.path.abspath(sys.argv[1])
train_dir = os.path.abspath(os.path.join(sys.argv[2], 'train'))
test_dir = os.path.abspath(os.path.join(sys.argv[2], 'test'))

usps_train = USPS(work_dir, train=True)
usps_test = USPS(work_dir, train=False)

start_idx = 0
start_idx += unpack(usps_train, train_dir, start_idx)

start_idx = 0
start_idx += unpack(usps_test, test_dir, start_idx)


# def random_sample(data_path, out_path, repeats, label_num):
#     subdirs = os.listdir(data_path)     
#     for i in range(0, repeats):
#         jdx = 0
#         for subdir in subdirs:
#             images = os.listdir(os.path.join(data_path, subdir)) 
#             idx = np.array([int(name.split('_')[0]) for name in images])
#             # print(idx)
#             np.random.shuffle(idx)
#             if label_num == 0:
#                 selected_idx = list(idx)
#             else:
#                 selected_idx = list(idx[:label_num])
#             for idx in selected_idx:
#                 image_name = '{}_{}.png'.format(idx, subdir)
#                 with open(os.path.join(out_path, '{}.txt'.format(i)), 'a') as f:
#                     f.write('{} {}\n'.format(image_name, subdir))
#                     jdx+=1
#         print(jdx)

# random_sample(train_dir, train_dir, 2, 7000)