#!/usr/bin/env python
# coding: utf-8

import os
import gc
import time
import pandas as pd
import numpy as np
import multiprocessing 
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from func import build_AlexNet, build_ResNet
from tensorflow.keras.optimizers import SGD


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("repeat_idx", type = int)
args = parser.parse_args()
idx = args.repeat_idx

IMSIZE = 128

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    '/database/datasets/Classics/CatDog/validation/',
    target_size=(IMSIZE, IMSIZE),
    batch_size = 1000,
    class_mode ='categorical')



path = '/mgd_experiment/'
logs = []

file_path = path + 'model/'
save_path = path + 'result/'
#file_list = os.listdir(file_path)


loss = []
acc = []
for k in range(80):
    if k<9:
        model_path = file_path + 'tf_weights_ {}_{}.h5'.format(k+1, idx)
    else:
        model_path = file_path + 'tf_weights_{}_{}.h5'.format(k+1, idx)
    print(model_path)
    model = build_AlexNet(imsize = 128, num_class = 2)
    model.load_weights(model_path)
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = SGD(learning_rate = 0.01, 
                                  momentum = 0.9, nesterov = True), metrics = ['accuracy'])
    result = model.evaluate(validation_generator, steps = 10)
    del model
    gc.collect()

    loss.append(result[0])
    acc.append(result[1])


data = {'val_loss': loss, 'val_acc': acc}
df = pd.DataFrame(data)
df.to_csv(save_path + 'tf_result_{}.csv'.format(idx))
