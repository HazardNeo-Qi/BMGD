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
niter = [4,4,5,10]
epoch = [10,5,2,1]
step = len(niter)

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    '/database/datasets/Classics/CatDog/validation/',
    target_size=(IMSIZE, IMSIZE),
    batch_size = 1000,
    class_mode ='categorical')



path = '/bmgd_experiment/'
logs = []

file_path = path + 'model/'
save_path = path + 'result/'
#file_list = os.listdir(file_path)


loss = []
acc = []
count = 0
for i in range(step):
    for j in range(niter[i]):
        count = count + 1
        for k in range(epoch[i]):
            
            if k<9:
                model_path = file_path + 'bmgd_weights_{}_ {}_{}.h5'.format(count,k+1, idx)
            else:
                model_path = file_path + 'bmgd_weights_{}_{}_{}.h5'.format(count,k+1, idx)
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
df.to_csv(save_path + 'bmgd_result_{}.csv'.format(idx))
