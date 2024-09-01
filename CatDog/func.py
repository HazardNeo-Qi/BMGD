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
from tensorflow.keras.layers import BatchNormalization,Conv2D,Dense,Flatten,Input,MaxPooling2D, Dropout, BatchNormalization



'''
读取图片 准备X,Y
'''
def img_reading(img_path, class_num, img_size, output_list):
    Img=Image.open(img_path)
    Img_RGB = Img.convert("RGB")
    Img_RGB_resized = Img_RGB.resize((img_size, img_size))
    img = np.array(Img_RGB_resized)
    output_list.append([img, class_num])


'''
多进程读取图片
'''
def mp_img_reading(file_list, img_size, num_class, scale):

    try:
        cpu_count = multiprocessing.cpu_count()
        mp_data_list = multiprocessing.Manager().list()
        args_list = [(file[0], file[1], img_size, mp_data_list) for file in file_list]

        pool = multiprocessing.Pool(processes = cpu_count)
        pool.starmap_async(img_reading, args_list)

    finally:
        pool.close()
        pool.join()

    img_arr_list = list(mp_data_list)
    img_arr = np.array(img_arr_list, dtype=object)
    x_list = [arr[0] for arr in img_arr_list]

    X = np.asarray(x_list) / scale
    Y = tf.keras.utils.to_categorical(img_arr[:, 1], num_classes=num_class)

    return X, Y


'''
实现CPU和GPU并行操作
'''

def data_prep(train_list, val_list, img_size, num_class, scale, buffer_size, big_batch_q):

    big_batch_train_length = int(ceil(len(train_list) / buffer_size))
    #big_batch_val_length = int(ceil(len(val_list) / buffer_size))
    #print("big_batch_train_length, big_batch_val_length: ", big_batch_train_length, big_batch_val_length)
    print("big_batch_train_length: ", big_batch_train_length)

    for i in range(big_batch_train_length):
        bigBatch_X_train, bigBatch_Y_train = mp_img_reading(train_list[i*buffer_size:(i+1)*buffer_size], img_size, num_class, scale)
        #bigBatch_X_val, bigBatch_Y_val = mp_img_reading(val_list[i*buffer_size:(i+1)*buffer_size], img_size, num_class, scale)

        if i == (big_batch_train_length-1):
            #big_batch_q.put([bigBatch_X_train, bigBatch_Y_train, bigBatch_X_val, bigBatch_Y_val, 0])
            big_batch_q.put([bigBatch_X_train, bigBatch_Y_train, 0])
        else:
            #big_batch_q.put((bigBatch_X_train, bigBatch_Y_train, bigBatch_X_val, bigBatch_Y_val, 1))
            big_batch_q.put([bigBatch_X_train, bigBatch_Y_train, 1])

def training(big_batch_q):

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

    model = build_AlexNet()
    model.compile(loss = 'categorical_crossentropy', optimizer = SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True), metrics = ['accuracy'])

    cycle_num = 0
    while True:
        if not big_batch_q.empty():
            cycle_num += 1
            #bigBatch_X_train, bigBatch_Y_train, bigBatch_X_val, bigBatch_Y_val, ending = big_batch_q.get()
            bigBatch_X_train, bigBatch_Y_train, ending = big_batch_q.get()
            #model.fit(bigBatch_X_train, bigBatch_Y_train, epochs = 5, batch_size = 50, validation_data = (bigBatch_X_val,bigBatch_Y_val))
            model.fit(bigBatch_X_train, bigBatch_Y_train, epochs = 5, batch_size = 100)
            gc.collect()
            print("Cycle {} finished.".format(cycle_num))
            if ending == 0:
                break
        else:
            sleep(1)


def build_AlexNet(imsize = 227, num_class = 1000, weight_decay = 1e-4):
    
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


def build_ResNet(imsize = 224, num_class = 1000):

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


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, master_file, imsize, batch_size = 32, n_classes = 1000, shuffle=True):
        
        'Initialization'
        self.master_file = pd.read_csv(master_file)
        self.file_list = list(self.master_file['Path'])
        self.label_dict = {self.master_file['Path'][i] : self.master_file['Class_Num'][i] for i in range(len(self.master_file['Path']))}
        
        self.imsize = imsize
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
        print("Found {} images !".format(len(self.file_list)))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.file_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.imsize, self.imsize, 3))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, fp in enumerate(list_IDs_temp):
            # Store sample
            Img = Image.open(fp)
            Img_RGB = Img.convert("RGB")
            Img_RGB_resized = Img_RGB.resize((self.imsize, self.imsize))
            img = np.array(Img_RGB_resized)
            X[i,] = img

            # Store class
            y[i] = self.label_dict[fp]

        return X, tf.keras.utils.to_categorical(y, num_classes = self.n_classes)
    
    






'''
模拟实验部分函数：
'''

'''
数据生成过程：
'''
def generate_data_Linear(N,p,beta, Sigma):
    
    x = np.random.normal(size = [N,p])
    R = np.linalg.cholesky(Sigma)
    X = np.matmul(x, R.T)
    eps = np.random.normal(size = [N,1])
    Y = np.matmul(X,beta_0) +eps
    
    return X, Y


def generate_data_Logistic(N,p,beta, Sigma):
    
    x = np.random.normal(size = [N,p])
    R = np.linalg.cholesky(Sigma)
    X = np.matmul(x, R.T)
    u = np.matmul(X, beta)
    prob = np.exp(u)
    prob = prob/(1+prob)
    y = np.random.uniform(size = [N,1])
    Y = 1. * (y<prob)
    
    return X, Y


def EntropyLoss(X,Y,beta):
    
    N = X.shape[0]
    u = np.matmul(X,beta)
    prob = np.exp(u)
    prob = prob/(1+prob)
    L = -Y*np.log(prob+1e-6) - (1-Y)*np.log(1-prob+1e-6)

    return np.sum(L)/N


def SquareLoss(X,Y,beta):
    N = X.shape[0]
    L = Y-np.matmul(X,beta) 
    return np.sum(L**2)/N


'''
定义单个epoch的MGD函数
'''

def MGD_Linear(X,Y, beta, batch_size = 10, learning_rate = 0.01):
    
    N,p = X.shape
    batch_num = int(np.ceil(N/batch_size))
    Bin = batch_num/30
    for i in range(batch_num):
        
        x = X[i*batch_size:(i+1)*batch_size]
        y = Y[i*batch_size:(i+1)*batch_size]
        grad = np.matmul(np.matmul(x.T,x)/batch_size, beta) - np.matmul(x.T,y)/batch_size
        update = - learning_rate*grad
        beta = beta + update
        loss = SquareLoss(X,Y,beta)
        dist = np.sqrt(np.sum(update**2))
        
        #print('\rBatch {:d}/{:d} ['.format(i+1, batch_num) + '='*int(i/Bin) + '>' + '.'*(29-int(i/Bin)) + '] - loss: {:.4f} - dist: {:.4f}'.format(loss,dist), end = '', flush = True)
    
    #print('\rBatch {:d}/{:d} ['.format(batch_num, batch_num) + '='*30 + '] - loss: {:.4f} - dist: {:.4f}'.format(loss, dist), end = '\n')
    
    return beta



def cal_grad(x,y, beta):
    
    n,p = x.shape
    u = np.matmul(x, beta)
    prob = np.exp(u)
    prob = prob/(1+prob)
    resid = y-prob
    grad = np.mean(x*resid, axis = 0).reshape([p,1])
    
    return -grad


def cal_hess(x,y, beta):
    
    n,p = x.shape
    u = np.matmul(x, beta)
    prob = np.exp(u)
    prob = prob/(1+prob)
    weight = np.sqrt(prob*(1-prob))
    wx = x*weight
    hess = np.matmul(wx.T, wx)/n
    return hess


'''
定义单个epoch的MGD函数
'''

def MGD_Logistic(X,Y, beta, batch_size = 10, learning_rate = 0.01):
    
    N,p = X.shape
    batch_num = int(np.ceil(N/batch_size))
    Bin = batch_num/30
    for i in range(batch_num):
        
        x = X[i*batch_size:(i+1)*batch_size]
        y = Y[i*batch_size:(i+1)*batch_size]
        grad = cal_grad(x,y, beta)
        update = - learning_rate*grad
        beta = beta + update
        loss = EntropyLoss(X,Y,beta)
        dist = np.sqrt(np.sum(update**2))
        
        print('\rBatch {:d}/{:d} ['.format(i+1, batch_num) + '='*int(i/Bin) + '>' + '.'*(29-int(i/Bin)) + '] - loss: {:.4f} - dist: {:.4f}'.format(loss,dist), end = '', flush = True)
    
    print('\rBatch {:d}/{:d} ['.format(batch_num, batch_num) + '='*30 + '] - loss: {:.4f} - dist: {:.4f}'.format(loss, dist), end = '\n')
    
    return beta


def MLE(X,Y, tol = 1e-8):
    
    n,p = X.shape
    dist = 1.0
    beta_hat = np.zeros([p,1])
    while (dist>tol):
        
        grad = cal_grad(X,Y,beta_hat)
        hess = cal_hess(X,Y,beta_hat)
        update = - np.matmul(np.linalg.inv(hess),grad)
        beta_hat = beta_hat + update
        loss = EntropyLoss(X,Y, beta_hat)
        dist = np.sum((update)**2)/p
        print(dist, loss)
        
    return beta_hat









