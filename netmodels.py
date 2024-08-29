#!/usr/bin/env python
# coding: utf-8

'''
Basic functions 
'''

import os
import gc
import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing 
from math import floor, ceil
from time import sleep, time


from keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization,Conv2D,Dense,Flatten,Input,MaxPooling2D, Dropout, BatchNormalization, Activation, add
from tensorflow.keras.layers import *


def update_par(k,v):
    try:
        return v
    except Exception as e:
        print(e)
        return k

def build_AlexNet( info ):
    imsize = 227
    num_class = 1000
    weight_decay = 1e-4
    
    imsize = update_par(imsize, info['im_size'])
    num_class = update_par(num_class, info['num_class'])
    
    # 输入层
    input_layer = Input([imsize,imsize,3])
    
    x = Conv2D(96,[11,11], strides = [4,4], activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(input_layer) 
    x = BatchNormalization()(x)
    x = MaxPooling2D([3,3], strides = [2,2])(x)    
    
    x = Conv2D(256,[5,5],padding = "same", activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D([3,3], strides = [2,2])(x)
    x = Conv2D(384,[3,3],padding = "same", activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x) 
    x = BatchNormalization()(x)
    x = Conv2D(384,[3,3],padding = "same", activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x) 
    x = BatchNormalization()(x)
    x = Conv2D(256,[3,3],padding = "same", activation = 'relu',
               kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(x) 
    x = BatchNormalization()(x)
    x = MaxPooling2D([3,3], strides = [2,2])(x)
    x = Flatten()(x)   
    x = Dense(4096,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_class,activation = 'softmax')(x) 
    model=Model(input_layer,output_layer)
    return model


def build_ResNet( info ):
    imsize = 224
    num_class = 1000
    
    imsize = update_par(imsize, info['im_size'])
    num_class = update_par(num_class, info['num_class'])
    
    input_layer = Input([imsize,imsize,3])
    #基本的数据预处理
    x = Conv2D(64,[7,7], padding = "same", strides = [2,2])(input_layer)                                                            #64个3x3-same类型卷积
    x = BatchNormalization()(x)                                                                          #Batch Normalization
    x = Activation('relu')(x)                                                                            #relu激活

    #第一个残差block
    x = MaxPooling2D([3,3], padding = "same",strides=[2,2])(x)                                           #最大池化：注意矢量尺寸不变
    x0 = x                                                                                               #copy一个x0张量
    for k in range(2):
        x = Conv2D(64,[3,3], padding = "same")(x);x = BatchNormalization()(x);x = Activation('relu')(x)  #卷积，BN标准化，relu激活
        x = Conv2D(64,[3,3], padding = "same")(x);x = BatchNormalization()(x)                            #同上，但是没有relu激活
        x = add([x,x0])                                                                                  #残差张量叠加
        x = Activation('relu')(x)                                                                        #relu激活
        x0 = x                                                                                           #更新x0

    ##接下来3*2=6个标准残差block
    for nblocks in range(3):

        depth=128*2**nblocks                                                                                     #张量通道持续增加
        x0 = Conv2D(depth,[3,3], padding = "same", strides = [2,2])(x0)                                          #张量尺寸持续缩小
        x0 = BatchNormalization()(x0);x0 = Activation('relu')(x0)                                                #BN正则化与激活
        x = x0                                                                                                   #x张量尺寸缩小

        for k in range(2):
            x = Conv2D(depth,[3,3], padding = "same")(x);x = BatchNormalization()(x);x = Activation('relu')(x)   #卷积，BN标准化，relu激活
            x = Conv2D(depth,[3,3], padding = "same")(x);x = BatchNormalization()(x)                             #同上，但是没有relu激活
            x = add([x,x0])                                                                                      #残差张量叠加
            x = Activation('relu')(x)                                                                            #relu激活
            x0 = x                                                                                               #copy一个x0张量


    ## gap-fc
    x = GlobalAveragePooling2D()(x)                                                                              #全局平均池化提取特征
    output_layer = Dense(num_class,activation = 'softmax')(x)                                                                      #全链接到输出层
                                                                                          
    model = Model(input_layer,output_layer)                                                                     #模型组装                                                                                         
    return model






