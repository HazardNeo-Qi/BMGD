import os,shutil,time,gc
import numpy as np
import multiprocessing
import pandas as pd


def img_reading(img_path_list, output_list):
    # 处理一个batch的图片数据
    # 输入一些batch的id，返回对应的X,Y（batch id 对应的图像之X,Y）
    X = []
    Y = []
    for _id in img_path_list:
        _X = np.load('/home/work/train/X/%d.npy' % _id).reshape([-1,512])
        _Y = np.load('/home/work/train/Y/%d.npy' % _id).reshape([-1,1])
        X.extend(_X)
        Y.extend(_Y)
    X = np.array(X)
    Y = np.array(Y) 
    z
    # 由于每个图片会产生64X64=4096个样本，其中只有个位数的样本为正样本
    # 所以进行降采样，快速平衡正负样本比例（先抽样，然后保留所有正样本）
    n_all = len(X)
    sample_ids = list(pd.Series(list(range(n_all))).sample( int(n_all*0.002), replace=False ))
    sample_ids = list(np.where(Y==1)[0]) + sample_ids
    sample_X = X[sample_ids]
    sample_Y = Y[sample_ids]
    
    output_list.append([sample_X, sample_Y])

    
def mp_img_reading_v2(idList_arr):
    # 多进程读取数据
    
    buffer_size = 5662
    n_thread = 10 # multiprocessing.cpu_count()
    batch_per_cpu = int(np.ceil(buffer_size/n_thread))+1 # 每个子进程处理的图片个数

    # 生成每个子进程处理的图片 id list
    data_list = []
    for i in range(n_thread):
        st = i*batch_per_cpu
        ed = min(st+batch_per_cpu, len(idList_arr)) # 最后一个batch读不了batch_size这么多的图片，读到最后一个图片即可
        sub_data_list = idList_arr[st:ed] # 记录第 i 个子进程应该读取的图片，为第 st~第 ed 个
        data_list.append(sub_data_list)
    # 公共列表，用于记录子进程返回的数据
    output_list = multiprocessing.Manager().list()

    # 多个子进程并发处理图像
    ps = []
    for i in range(n_thread):
        p = multiprocessing.Process(target=img_reading, args=(data_list[i], output_list,))
        ps.append(p)
    for i in range(n_thread):
        ps[i].start()
    for i in range(n_thread):
        ps[i].join()
    
    # 拼装多个子进程返回的结果
    img_arr_list = list(output_list)
    x_list = [each[0] for each in img_arr_list]
    y_list = [each[1] for each in img_arr_list]
    x=np.vstack(x_list)
    y=np.vstack(y_list)
 
    return x, y



def epoch_train(j, k, model_input):
    # 将一次buffer训练的过程封装为进程
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model  
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.callbacks import ModelCheckpoint
    from func import MyModel
    
    _X = model_input['X'] # buffer - X
    _Y = model_input['Y'] # buffer - Y
    tc_buffer_read = model_input['tc_buffer'] # buffer 读取时间，后面直接记录
    _times = model_input['times'] # buffer上训练轮数
    _batch_size = model_input['batch_size'] # buffer上训练的 batch size
    save_path = model_input['save_path'] # 模型保存位置
    res_path = model_input['res_path'] # 运行时间记录位置
    rt = model_input['rt'] # 重复实验的次数
    
    '''
    # 多gpu
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
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
    '''
    _t0 = time.time()
    # 声明模型
    model = MyModel()
    # 第二次开始，读取保存的模型，在前一次训练好的模型基础上做训练更新
    if j==0 and k==0:
        model.save_weights(save_path + 'model_b/' + str(rt) + 'init.h5')
    else:
        model.load_weights('tmp_weight_b.h5')
    model.compile(loss='binary_crossentropy',optimizer = Adam(learning_rate = 0.001), metrics=['accuracy'])
    _t1 = time.time()
    # 模型读取时间
    tc_model_load = _t1 - _t0

    # 训练过程中的模型保存
    checkpoint = ModelCheckpoint(save_path + 'model_b/' + str(rt) + 'bmgd_weights_' + str(j)+ '-' + str(k) + '_{epoch:2d}.h5',
                               verbose = 0, save_weights_only = True, save_freq = 'epoch')

    _t0 = time.time()
    # 一个buffer上的训练
    model.fit(_X, _Y, epochs = _times, batch_size = _batch_size, callbacks = [checkpoint], verbose=1)
    _t1 = time.time()
    # 一个buffer上模型训练的时间
    tc_model_train = _t1 - _t0 

    # 将模型保存起来
    _t0 = time.time()
    model.save_weights('tmp_weight_b.h5')
    model.save_weights(save_path + 'model_b/' + str(rt) + '=' + str(j)+ '-' + str(k) + '.h5')
    _t1 = time.time()
    # 模型保存时间
    tc_model_save = _t1 - _t0 

    # 记录训练时间
    with open(res_path, 'a') as f:
        f.write('%d,%d,%d,%.4f,%.4f,%.4f,%.4f\n' % 
                (j, k, _times, tc_buffer_read, tc_model_load, tc_model_train, tc_model_save))


if __name__ == '__main__':
    
    """
    1 创建文件夹
    """
    save_path = '/mnt/TTChina/result_B0324/'

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    os.mkdir(save_path + 'model_b/')
    os.mkdir(save_path + 'result_b/')
    print('结果保存位置： \"', save_path, '\"')
    
    # 实验重复10次
    for rt in range(10):
        np.random.seed(rt)
        
        """
        2 准备工作 
        """
        # 记录模型运行的时间
        res_path = save_path + 'result_b/timecost_' + str(rt) + '.csv'
        with open(res_path, 'w') as f:
            f.write('step,niter,epochs,tc_buffer,tc_load,tc_train,tc_save\n')   
        n_train = 5662 # 训练集个数

        """
        3 开始训练
        """
        idList_arr = list(range(n_train))
        niter = np.array([4,4,5,10])
        epoch = np.array([10,5,2,1])
        step = 4

        for j in range(step):

            for k in range(niter[j]):
                # 打乱图片id列表顺序
                np.random.shuffle(idList_arr)
                s1 = time.time()
                # 读取一个buffer，返回的图片为X0、标注为Y0
                X0, Y0 = mp_img_reading_v2(idList_arr)
                s2 = time.time()
                # 读取一个buffer数据时间
                tc_buffer_read = s2-s1

                model_input = {
                    'X': X0,
                    'Y': Y0,
                    'tc_buffer': tc_buffer_read,
                    'times': epoch[j], # 当前buffer上训练多少轮
                    'batch_size': 1000, # 当前buffer上训练时的 batch_size （一个batch处理多少行样本）。和 Default.py 中不同（Default.py中是batch_size是指处理多少个图像。Default.py中若batch_size设置为150，那么对应buffer方法下的batch_size大概是1228.8，这里取整一点的数值，用1000）。
                    'save_path': save_path,
                    'res_path': res_path,
                    'rt': rt
                }
                # 将buffer的训练封装为进程
                p = multiprocessing.Process(target=epoch_train, args=(j, k, model_input,))
                p.start()
                p.join() # 进程结束后，内存、显存会自动释放

                print('* [%d]- iteration %d-%d OK!' % (rt,j,k))



        
