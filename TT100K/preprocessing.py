import os
import re
import numpy as np
import time
from matplotlib import pyplot as plt
from PIL import Image
import shutil
import json

"""
1.整理数据
"""
dx = 2048
dy = 2048
dz = 3

data_dir = '/database/datasets/TS/ChinaTS/' # 训练数据目录

def get_ids(data_dir, op):
    # 读取数据集的图片列表
    t0 = time.time()
    samples = [] # 记录所有图片的ID
    with open(data_dir + op + '/ids.txt', 'r') as f:
        line = ' '
        while line:
            line = f.readline()
            _name = line.strip()
            if len(_name)>0:
                samples.append(_name)
    _n = len(samples)
    t1 = time.time()
    print('%d %s 图像. %.2fs' % (_n, op, t1-t0))
    return samples, _n

# 训练集
samples_train, n_train = get_ids(data_dir, 'train')
samples_test, n_test = get_ids(data_dir, 'test')

# 读取标注
with open('/database/datasets/TS/ChinaTS/annotations.json', 'r') as f:
    js = json.loads(f.read())
print('标注JSON读取OK')


# 筛选数据
def myfilter(_samples, _tag):
    new_sample = []
    for _name in _samples:
        _objects = js['imgs'][_name]['objects']
        for _object in _objects:
            if _tag in _object['category']:
                new_sample.append(_name)
                break
    return new_sample

samples_train = myfilter(samples_train, 'p')
samples_test = myfilter(samples_test, 'p')
n_train = len(samples_train)
n_test = len(samples_test)
print('%d train 图像.' % n_train)
print('%d test 图像.' % n_test)

'''
3.VGG16模型，模型迁移，得到特征图
'''
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,AveragePooling2D,Reshape, MaxPooling2D


base_model = VGG16(weights='imagenet', input_shape = [dx,dy,3], include_top = False)
x = base_model.output
#x = MaxPooling2D([2,2])(x)
xmodel = Model(inputs=base_model.input, outputs=x)
for layer in base_model.layers:
    layer.trainable = False
xmodel.summary()

"""
4.对因变量也利用模型变形
"""

input_layer = Input([dx,dy,1])                             # Input: 960*1280*1 Y
x = input_layer                                            # CNN model: Input layer
x = AveragePooling2D([32,32])(x)                           # CNN model: Average pooling to down-sampling
x = Reshape([64,64])(x)                                    # Reshape to drop the redundant dimension
output_layer = x                                           # CNN model: Output layer
ymodel = Model(input_layer,output_layer)                   # CNN model: Combine
ymodel.summary()                                           # CNN model: Summarize the structure of the model

"""
5.数据存储到硬盘
"""
# Process all the image to generate X and Y
if os.path.exists('/home/work'): 
    shutil.rmtree('/home/work')     #如果这个目录已经存在，那么彻底删除
os.mkdir('/home/work/')
os.mkdir('/home/work/train/') 
os.mkdir('/home/work/train/X')                                    
os.mkdir('/home/work/train/Y')                                 
os.mkdir('/home/work/test/') 
os.mkdir('/home/work/test/X')                                    
os.mkdir('/home/work/test/Y')  


"""
2.数据张量化
""" 
def sample_to_XY(_samples, op):
    # 将samples变成X、Y数据
    global dx, dy, dz
    global xmodel, ymodel, js

    N = len(_samples)#25 # 样本数
    print('开始: %s. %d' % (op, N))
                       
    t0 = time.time()
    for i in range(N): 
        _t0 = time.time()
        # 产生空样本
        X = np.zeros([1,dx,dy,dz])
        Y = np.zeros([1,dx,dy])   

        # 文件名
        _name = _samples[i]
        # 读取图像
        _img = Image.open(data_dir+op+'/'+_name+'.jpg')
        _img = np.array(_img) / 255
        X[0] = _img
        # Xmodel处理
        fmap = xmodel.predict(X)                     
        np.save('/home/work/'+op+'/X/'+str(i)+'.npy', fmap.reshape([64,64,512])) 

        # 标记框
        _tags = js['imgs'][_name]['objects']
        for _tag in _tags:
            # 如果是标识牌
            if 'p' in _tag['category']:
                # 则获取bbox
                _box = _tag['bbox']
                _ymin = int(_box['xmin'])
                _xmin = int(_box['ymin'])
                _ymax = int(_box['xmax'])
                _xmax = int(_box['ymax'])   
                # 标记，该区域为bbox内容
                Y[0, _xmin:_xmax, _ymin:_ymax] = 1
        # Ymodel处理
        YY_train = ymodel.predict(Y.reshape([1,dx,dy,1])) 
        _t1 = time.time()
        if i % 50 == 0:
            print('%s - %d / %d. (%.2f) %.2fs' % (op, i, N, YY_train.max(), _t1-_t0))
        YY_train = 1.0*(YY_train>0.2)  
        np.save('/home/work/'+op+'/Y/'+str(i)+'.npy', YY_train.reshape([64,64]))  

        
    t1 = time.time()            
    print('%s - X-Y 样本产生完成. %d 个样本. %.2fs' % (op, N, t1-t0))
    return X,Y

sample_to_XY(samples_train, 'train')
sample_to_XY(samples_test, 'test')
print('OK')


