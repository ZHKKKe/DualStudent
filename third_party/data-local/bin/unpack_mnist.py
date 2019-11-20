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

from tqdm import tqdm
from torchvision.datasets import MNIST
import matplotlib.image
import numpy as np


work_dir = os.path.abspath(sys.argv[1])
train_dir = os.path.abspath(os.path.join(sys.argv[2], 'train'))
test_dir = os.path.abspath(os.path.join(sys.argv[2], 'test'))

mnist_train = MNIST(work_dir, train=True, download=True)
mnist_test = MNIST(work_dir, train=False, download=True)


def unpack(source_data, target_dir, start_idx):
    for idx, (image_data, label_idx) in tqdm(enumerate(source_data), total=len(source_data)):
        subdir = os.path.join(target_dir, str(label_idx))
        name = '{}_{}.png'.format(start_idx + idx, str(label_idx))
        os.makedirs(subdir, exist_ok=True)
        image_data = image_data.convert('L')
        image_data.save(os.path.join(subdir, name))
    return len(source_data)


start_idx = 0
start_idx += unpack(mnist_train, train_dir, start_idx)

start_idx = 0
start_idx += unpack(mnist_test, test_dir, start_idx)

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
