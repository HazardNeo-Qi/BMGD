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
import shutil
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
save_path = '/bmgd_experiment/'

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

        
# 图片的list信息
FILE_LIST_PATH = '/mnt/CatDog_train.csv'
idList = pd.read_csv(FILE_LIST_PATH, sep=',')
idList_arr = np.array(idList)


def img_reading(img_path_list, output_list):
    
    # 预备一个空的array
    n = len(img_path_list)
    X = np.zeros([n,128,128,3])
    y = np.zeros([n])
    
    # for循环读取数据
    for i, img_path in enumerate(img_path_list):
          
        Img=Image.open(img_path[0])
        Img_RGB = Img.convert("RGB")
        Img_RGB_resized = Img_RGB.resize((128,128))
        img = np.array(Img_RGB_resized)
        X[i] = img
        y[i] = img_path[1]
    Y = tf.keras.utils.to_categorical(y, num_classes = 2)
    
    # 用事先准备好的生成器数据增强
    train_aug = train.flow(X,Y, batch_size = n)
    X_aug,Y_aug = next(train_aug)
    output_list.append([X_aug,Y_aug])
    
    
    
train = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.5,
        rotation_range=30,
        zoom_range=0.2, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

def mp_img_reading_v2(file_list):

    try:
        cpu_count = multiprocessing.cpu_count()

        batch_per_cpu = int(np.ceil(15000/cpu_count))
        # 生成每个GPU读取的list
        data_list = []
        for i in range(cpu_count):
            sub_data_list = idList_arr[i*batch_per_cpu:(i+1)*batch_per_cpu]
            data_list.append(sub_data_list)

        output_list = Manager().list()
        #args_list=list(range(cpu_count))
        args_list=[(each,output_list) for each in data_list]
        pool = Pool(processes = cpu_count)
        pool.starmap_async(img_reading,args_list)    

    finally:
        pool.close()
        pool.join()

    img_arr_list = list(output_list)
    x_list = [each[0] for each in img_arr_list]
    y_list = [each[1] for each in img_arr_list]
    x=np.vstack(x_list)
    y=np.vstack(y_list)
    return x, y


niter = np.array([4,4,5,10])
epoch = np.array([10,5,2,1])
step = 4
train_loss = []
train_acc = []

Time = np.zeros([3, sum(niter)])
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    s1=time.time()
    model=build_AlexNet(imsize = 128, num_class = 2)
    #model.load_weights(model_path)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate = 0.001),metrics=['accuracy'])
    s2=time.time()
print('initializaton: ',s2-s1)



count = 0
for j in range(step):
    
    if j==(step-1):
        model.optimizer.learning_rate = 0.0001
    
    for n in range(niter[j]):
        
        count += 1
        checkpoint = ModelCheckpoint(save_path + 'model/' + 'bmgd_weights_' + str(count) + '_{epoch:2d}_' + str(idx)+'.h5',
                                   verbose = 1, save_weights_only = True, save_freq = 'epoch')

        

        np.random.shuffle(idList_arr)
        s1 = time.time()
        X0, Y0 = mp_img_reading_v2(file_list = idList_arr)
        s2 = time.time()
        print(s2-s1)
        history = model.fit(X0, Y0, epochs = epoch[j],batch_size = 150*5, callbacks = [checkpoint])
        s3 = time.time()

        del X0,Y0
        gc.collect()
        s4 = time.time()
        Time[0,n] = s2 - s1
        Time[1,n] = s3 - s2
        Time[2,n] = s4 - s1

        train_loss.extend(history.history['loss'])
        train_acc.extend(history.history['accuracy'])
        

times1 = np.ones([np.sum(niter*epoch)])*np.mean(Time[0])
times2 = np.ones([np.sum(niter*epoch)])*np.mean(Time[1])

# 保存其他信息
logs = {'data_time':np.cumsum(times1), 'time':np.cumsum(times2), 'train_loss':train_loss, 'train_acc': train_acc}
df = pd.DataFrame(logs)
df.to_csv(save_path + 'result/'+'BMGD_training_result_'+str(idx)+'.csv', index = False)
        
        
