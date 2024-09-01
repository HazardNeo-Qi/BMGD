#!/usr/bin/env python
# coding: utf-8

import os,gc,shutil,IPython,time,multiprocessing
import numpy as np
import pandas as pd
from PIL import Image
from math import floor, ceil
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization,Conv2D,Dense,Flatten,Input,MaxPooling2D, Dropout, BatchNormalization
from func import build_AlexNet, build_ResNet
from keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K

from multiprocessing import Process,Manager,Pool
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import multiprocessing
import numpy as np
import pandas as pd
from PIL import Image
import argparse

# 传递参数 方便重复实验
parser = argparse.ArgumentParser(description='命令行中传入一个数字')
parser.add_argument('integers', type=int, help='传入的数字')
args = parser.parse_args()
idx = args.integers

# 创建文件夹
save_path = '/mgd_experiment/'

if os.path.exists(save_path):
    shutil.rmtree(save_path)
    os.mkdir(save_path)
    os.mkdir(save_path + 'model/')
    os.mkdir(save_path + 'result/')
else:
    os.mkdir(save_path)
    os.mkdir(save_path + 'model/')
    os.mkdir(save_path + 'result/')

print('All results are saved in \"', save_path, '\"')


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

        
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMSIZE = 128

train = ImageDataGenerator(
    rescale=1./255,
        shear_range=0.5,
        rotation_range=30,
        zoom_range=0.2,   
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True).flow_from_directory(
    '/database/datasets/Classics/CatDog/train',
    target_size=(IMSIZE, IMSIZE),
    batch_size = 150*5,
    class_mode='categorical')

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    
    model=build_AlexNet(imsize = 128, num_class = 2)
    model.compile(loss='categorical_crossentropy',optimizer = Adam(learning_rate = 0.001),metrics=['accuracy'])

    
'''
keras自带调整学习率
'''
def scheduler(epoch):
    
    if epoch>69:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, 0.0001)
        print("lr changed to {}".format(0.0001))

    return K.get_value(model.optimizer.lr)

checkpoint = ModelCheckpoint(os.path.join(save_path + 'model/', 'tf_weights_{epoch:2d}_' + str(idx)+ '.h5'),
                                   verbose=1, save_weights_only = True, save_freq = 'epoch')
reduce_lr = LearningRateScheduler(scheduler)

history = model.fit(train,epochs = 80, callbacks = [checkpoint, reduce_lr], workers = 30)
s2=time.time()
print('timecost',s2-s1)



# 保存其他信息
logs = {'train_loss':history.history['loss'], 'train_acc': history.history['accuracy']}
df = pd.DataFrame(logs)
df.to_csv(save_path + 'result/'+'tf_training_result_' + str(idx)+'.csv', index = False)
        
        
