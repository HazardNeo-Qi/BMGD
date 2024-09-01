import os,shutil,time,gc
import numpy as np
import multiprocessing
import pandas as pd

def read_batch(batch_ids):
    # 读取一个batch的数据
    # 输入一些batch的id，返回对应的X,Y（batch id 对应的图像之X,Y）
    X = []
    Y = []
    for _id in batch_ids:
        _X = np.load('/home/work/train/X/%d.npy' % _id).reshape([-1,512])
        _Y = np.load('/home/work/train/Y/%d.npy' % _id).reshape([-1])
        X.extend(_X)
        Y.extend(_Y)
    X = np.array(X)
    Y = np.array(Y)
    
    # 由于每个图片会产生64X64=4096个样本，其中只有个位数的样本为正样本
    # 所以进行降采样，快速平衡正负样本比例（先抽样，然后保留所有正样本）
    n_all = len(X)
    sample_ids = list(pd.Series(list(range(n_all))).sample( int(n_all*0.002), replace=False ))
    sample_ids = list(np.where(Y==1)[0]) + sample_ids
    sample_X = X[sample_ids]
    sample_Y = Y[sample_ids]
    
    return sample_X, sample_Y


def epoch_train(i, model_input):
    # 将一个epoch的训练过程封装为进程
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model  
    from tensorflow.keras.optimizers import Adam, SGD
    from func import MyModel
    
    # 传入的参数：batch_num - batch个数；batch_id_dict - 每个batch读取的图片id； save_path - 模型保存位置
    batch_num = model_input['batch_num']
    batch_id_dict = model_input['batch_id_dict']
    save_path = model_input['save_path']
    res_path = model_input['res_path']
    rt = model_input['rt']
    
    '''
    # 当采用多gpu时需要的代码
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
    
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    '''

    _t0 = time.time()
    # 声明模型实例
    model = MyModel()
    # 第二次开始，读取保存的模型，在前一次训练好的模型基础上做训练更新
    if i > 0:
        model.load_weights('tmp_weight_t.h5')
    else:
        model.save_weights(save_path + 'model_t/' + str(rt) + 'init.h5')
    model.compile(loss='binary_crossentropy',optimizer = Adam(learning_rate = 0.001), metrics=['accuracy'])
    _t1 = time.time()
    # 模型读取时间
    tc_model_load = _t1 - _t0

    # 进行一个epoch的训练
    _t0 = time.time()
    # 遍历每个batch
    for _t in range(batch_num):
        # 读取一个batch数据
        X_batch, Y_batch = read_batch(batch_id_dict[_t])
        # 一个batch的训练
        model.fit(X_batch, Y_batch, epochs=1, batch_size = len(X_batch), verbose=0)
        del X_batch
        del Y_batch
        gc.collect()
        print(rt, '=', i, ':', _t)
    _t1 = time.time()
    # 模型训练时间
    tc_model_train = _t1 - _t0     

    # 将模型保存起来，后续再做效果检验
    _t0 = time.time()
    model.save_weights('tmp_weight_t.h5')
    model.save_weights(save_path + 'model_t/' + str(rt) + '=' + str(i) + '.h5')
    _t1 = time.time()
    # 模型保存的时间
    tc_model_save = _t1 - _t0 

    # 记录模型读取、训练、保存的时间
    with open(res_path, 'a') as f:
        f.write('%d,%.4f,%.4f,%.4f\n' % (i, tc_model_load, tc_model_train, tc_model_save))


if __name__ == '__main__':
    
    """
    1 创建文件夹
    """
    save_path = '/mnt/TTChina/result_0324/'
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    os.mkdir(save_path + 'model_t/')
    os.mkdir(save_path + 'result_t/')
    print('结果保存位置： \"', save_path, '\"')
    
    # 实验重复10次
    for rt in range(10):
        np.random.seed(rt)
        
        """
        2 准备工作 ，设置好每个batch要读的图片id
        """
        # 记录模型运行的时间
        res_path = save_path + 'result_t/timecost_' + str(rt) + '.csv'
        with open(res_path, 'w') as f:
            f.write('epoch,tc_load,tc_train,tc_save\n')

        # 训练集图片个数
        n_train = 5662 

        # 先组织好所有batch_id
        ids = list(range(n_train))
        # 打乱图片id顺序
        np.random.shuffle(ids) 

        batch_size = 150#*5
        batch_num = int(n_train / batch_size) + 1 
        batch_id_dict = {}
        for i in range(batch_num):
            st = i * batch_size 
            ed = min(st+batch_size, n_train) # 最后一个batch读不了batch_size这么多的图片，读到最后一个图片即可
            batch_id_dict[i] = ids[st:ed] # 记录第 i 个batch应该读取的图片，为第 st~第 ed 个
        # 封装成字典
        model_input = {
            'batch_num': batch_num, # batch个数
            'batch_id_dict': batch_id_dict, # 每个batch对应的id
            'save_path': save_path, # 模型保存位置
            'res_path': res_path, # 运行时间记录位置
            'rt': rt # 重复实验的次数
        }
        print("batch 列表完成")

        """
        3 开始训练
        """
        epochs = 50
        for i in range(epochs):
            # 将一个epoch的训练封装为进程进行
            p = multiprocessing.Process(target=epoch_train, args=(i, model_input,))
            p.start()
            p.join() # 进程结束后，内存、显存会自动释放

            print('* [%d] epoch - %d OK!' % (rt, i))

        
        
        
