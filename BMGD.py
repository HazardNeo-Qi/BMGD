import gc
import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing 
from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing.managers import SharedMemoryManager
from math import floor, ceil
from time import sleep, time
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization,Conv2D,Dense,Flatten,Input,MaxPooling2D, Dropout, BatchNormalization



class BMGD_worker(object):
    
    '''
    初始化，设置超参数
    '''
    def __init__(self, model_build=None, model_build_info=None, loss='categorical_crossentropy', init_lr=0.1,
                 epoch=10, iteration=10, batch_size=100, buffer_size=100,
                 save_path='bmgd_save/', img_size=128, channel_size=3, num_class=2, scale=255,
                 train_file_list='', val_file_list=''):
        
        # 模型创建方法
        self.model_build = model_build
        # 模型参数
        self.model_build_info = model_build_info
        # 损失函数
        self.loss = loss
        # 初始学习率
        self.init_lr = init_lr
        
        # buffer内循环
        self.epoch = epoch 
        # buffer外循环
        self.iteration  = iteration
        
        # batch size
        self.batch_size = batch_size
        # buffer size
        self.buffer_size = buffer_size

        # model save path 保存路径
        self.save_path = save_path

        # img 输入大小
        self.img_size = img_size
        # 通道数
        self.channel_size = channel_size
        # 分类类别数   
        self.num_class = num_class
        # 像素范围
        self.scale = scale

        # 训练数据路径名
        self.train_file_list = train_file_list
        # 验证数据路径名
        self.val_file_list = val_file_list
        
        # 是否记录所有checkpoints的模型
        self.save_all_checkpoints = False
        # 是否图像增强
        self.augment = False
        # 是否记录时间
        self.record = False
      

    '''
    读取图片 准备X,Y
    '''
    def img_reading(self, img_path, class_num, img_size, output_list):
        Img=Image.open(img_path)
        Img_RGB = Img.convert("RGB")
        Img_RGB_resized = Img_RGB.resize((img_size, img_size))
        img = np.array(Img_RGB_resized, dtype=np.float32)
        output_list.append([img, class_num])


    '''
    多进程读取图片
    '''
    def mp_img_reading(self, file_list, img_size, num_class, scale):

        try:
            cpu_count = multiprocessing.cpu_count()
            mp_data_list = multiprocessing.Manager().list()
            args_list = [(file[0], file[1], img_size, mp_data_list) for file in file_list]

            pool = multiprocessing.Pool(processes = cpu_count)
            pool.starmap_async(self.img_reading, args_list)
        except Exception as e:
            print(e)

        finally:
            pool.close()
            pool.join()

        time_st = time()
        img_arr_list = list(mp_data_list)
        img_arr = np.array(img_arr_list, dtype=object)
        x_list = [arr[0] for arr in img_arr_list]

        X = np.asarray(x_list) / scale
        Y = tf.keras.utils.to_categorical(img_arr[:, 1], num_classes=num_class)

        time_ed = time()

        del mp_data_list, args_list, img_arr_list, img_arr
        print('slot preparation time', time_ed - time_st)

        return X, Y

    '''
    多进程训练
    '''
    def training(self, writer_info_q, reader_info_q, shared_data, buffer_num, num_class, batch_size, epoch, reminder_buffer_size, path):
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
            
            #model = build_ResNet(imsize = 224, num_class = num_class)
            #model.compile(loss = 'categorical_crossentropy', optimizer = SGD(learning_rate = 0.02, momentum = 0.9, nesterov = True), metrics = ['accuracy'])

            model = self.model_build( self.model_build_info )
            model.compile(loss=self.loss, optimizer = SGD(learning_rate = self.init_lr), metrics=['accuracy'])
           
        
        if reminder_buffer_size == 0:
            AX, AY, BX, BY = shared_data
        else:
            AX, AY, BX, BY, RX, RY = shared_data

        # 若采用图像增强
        if self.augment:
            traings = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.5,
                rotation_range=30,
                zoom_range=0.2, 
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)
        else:
            traings = ImageDataGenerator()
   
        logs = []
        cycle_num = 0
        tic = time()
        while True:
            if not writer_info_q.empty():

                # 计算当前的iteration
                current_iter = floor(cycle_num/buffer_num)
                # 计算当前的buffer
                current_buffer = cycle_num%buffer_num
                
                slot_num, ending = writer_info_q.get()
                if slot_num == 0:
                    train_generator = traings.flow(AX, AY, batch_size = batch_size*len(gpus))
                elif slot_num == 1000: 
                    train_generator = traings.flow(RX, RY, batch_size = batch_size*len(gpus))
                else:
                    train_generator = traings.flow(BX, BY, batch_size = batch_size*len(gpus))
                
                if current_iter<3:
                    model.optimizer.learning_rate = self.init_lr
                
                elif current_iter<6:
                    model.optimizer.learning_rate = self.init_lr * 0.7
                
                elif current_iter<8:
                    model.optimizer.learning_rate = self.init_lr * 0.3
                
                else:
                    model.optimizer.learning_rate = self.init_lr * 0.1
                
                toc1 = time()
                model.fit(train_generator, epochs = epoch, workers=60)
                toc2 = time()
                if slot_num != 1000:
                    reader_info_q.put(slot_num)
                
                del train_generator
                gc.collect()
                # reader_info_q.put(slot_num)
                
                
                print("Current iteration: {}, {} buffer finished.".format(current_iter + 1, current_buffer +1 ))
                # 保存模型(单独计算验证集精度)
                if self.save_all_checkpoints:
                    model.save(path + 'model/iter{}_buffer{}.h5'.format(current_iter + 1, current_buffer +1))
                    
                # 记录模型运行时间
                logs.append([toc1-tic, toc2-toc1])
                if self.record:
                    df = pd.DataFrame(logs, columns = ['data_time', 'training_time'])
                    df.to_csv(path + 'result/iter{}_buffer{}.csv'.format(current_iter + 1, current_buffer +1), index = False)
                cycle_num += 1
                if ending == 0:
                    break
            else:
                print('writer queue is empty')
                sleep(10)
                
        final_save_path = path + 'model/iter{}_buffer{}.h5'.format(current_iter + 1, current_buffer +1)
        print('training is over. Model is saved at: [%s]' % final_save_path)
        # 记录最终模型
        model.save(final_save_path)


    '''
    实现CPU和GPU并行操作
    '''
    def data_prep(self, train_list, val_list, reminder_buffer_size, img_size, num_class, scale, buffer_size, iteration, writer_info_q, reader_info_q, shared_data):

        big_batch_train_length = int(ceil(len(train_list) / buffer_size)) 
        #big_batch_val_length = int(ceil(len(val_list) / buffer_size))
        #print("big_batch_train_length, big_batch_val_length: ", big_batch_train_length, big_batch_val_length)
        print("big_batch_train_length: ", big_batch_train_length)
        
        indices = np.arange(len(train_list))
        np.random.shuffle(indices)

        if reminder_buffer_size == 0:
            AX, AY, BX, BY = shared_data
        else:
            AX, AY, BX, BY, RX, RY = shared_data


        bigBatch_X_train, bigBatch_Y_train = self.mp_img_reading(train_list[indices[0:1*buffer_size]], img_size, num_class, scale)
        AX[:], AY[:] = bigBatch_X_train[:], bigBatch_Y_train[:]
        writer_info_q.put((0, 1))

        for i in range(1, iteration*big_batch_train_length):
            
            gc.collect()
            # 计算buffer的序号
            k = i%big_batch_train_length

            while reader_info_q.empty():
                sleep(1)
            
            # 确定要覆盖那一块空间
            need_to_cover_slot_num = reader_info_q.get()
            print('need_to_cover_slot_num: ', need_to_cover_slot_num)
            bigBatch_X_train, bigBatch_Y_train = self.mp_img_reading(train_list[indices[k*buffer_size:(k+1)*buffer_size]], img_size, num_class, scale)
            #bigBatch_X_val, bigBatch_Y_val = mp_img_reading(val_list[i*buffer_size:(i+1)*buffer_size], img_size, num_class, scale)

            if (k+1) == big_batch_train_length and reminder_buffer_size != 0:
                print("reminder work start!")
                reader_info_q.put(need_to_cover_slot_num)
                RX[:], RY[:] = bigBatch_X_train[:], bigBatch_Y_train[:]
                del bigBatch_X_train, bigBatch_Y_train
                if i == (iteration*big_batch_train_length-1):
                    writer_info_q.put((1000, 0))
                else:
                    writer_info_q.put((1000, 1))
            else:
                if need_to_cover_slot_num == 0:
                    AX[:], AY[:] = bigBatch_X_train[:], bigBatch_Y_train[:]
                else:
                    BX[:], BY[:] = bigBatch_X_train[:], bigBatch_Y_train[:]

                del bigBatch_X_train, bigBatch_Y_train

                if i == (iteration*big_batch_train_length-1):
                    writer_info_q.put((need_to_cover_slot_num, 0))
                else:
                    writer_info_q.put((need_to_cover_slot_num, 1))

    def main(self):
        # 获取超参数
        FILE_LIST_PATH_TRAIN = self.train_file_list
        FILE_LIST_PATH_VAL = self.val_file_list
        # READ FILE LIST
        idList_train = pd.read_csv(FILE_LIST_PATH_TRAIN, sep=',')
        idList_val = pd.read_csv(FILE_LIST_PATH_VAL, sep=',')
        idList_train_arr = np.array(idList_train)
        idList_val_arr = np.array(idList_val)
        img_size = self.img_size
        channel_size = self.channel_size
        num_class = self.num_class
        scale = self.scale
        batch_size = self.batch_size
        buffer_size = self.buffer_size
        
        
        # 创建文件夹，储存结果
        save_path = self.save_path

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
        
        
        float_precision = np.float32
        if float_precision == np.float32:
            nbytes_size = 4
        elif float_precision == np.float64:
            nbytes_size = 8
        reminder_buffer_size = len(idList_train_arr) % buffer_size
        
        # buffer内的循环
        epoch = self.epoch
        # buffer外循环
        iteration = self.iteration 
        
        # 计算存储空间的大小
        x_array_nbytes = buffer_size*img_size*img_size*channel_size*nbytes_size # for float32
        x_array_shape = (buffer_size, img_size, img_size, channel_size)
        y_array_nbytes = buffer_size*num_class*nbytes_size # for float32
        y_array_shape = (buffer_size, num_class)    
        
        # 最后一个buffer 单独处理
        if reminder_buffer_size != 0:
            x_reminder_nbytes = reminder_buffer_size*img_size*img_size*channel_size*nbytes_size # for float32
            x_reminder_shape = (reminder_buffer_size, img_size, img_size, channel_size)
            y_reminder_nbytes = reminder_buffer_size*num_class*nbytes_size # for float32
            y_reminder_shape = (reminder_buffer_size, num_class)   
        
        buffer_num = int(ceil(len(idList_train_arr)/buffer_size))
        
        # q = multiprocessing.Queue(q_size)
        # 两个消息队列，一个是写入完成的消息；一个是读入完成的消息，并且预置一次消息；
        writer_info_q = multiprocessing.Queue() # 数据处理完了 就加消息
        reader_info_q = multiprocessing.Queue() # 数据用完了 就加消息
        reader_info_q.put(1)

        with SharedMemoryManager() as smm:
            shm_X_A = smm.SharedMemory(size=x_array_nbytes)
            shm_Y_A = smm.SharedMemory(size=y_array_nbytes)
            AX = np.ndarray(x_array_shape, dtype=float_precision, buffer=shm_X_A.buf)
            AY = np.ndarray(y_array_shape, dtype=float_precision, buffer=shm_Y_A.buf)
            shm_X_B = smm.SharedMemory(size=x_array_nbytes)
            shm_Y_B = smm.SharedMemory(size=y_array_nbytes)
            BX = np.ndarray(x_array_shape, dtype=float_precision, buffer=shm_X_B.buf)
            BY = np.ndarray(y_array_shape, dtype=float_precision, buffer=shm_Y_B.buf)

            if reminder_buffer_size != 0:
                shm_X_R = smm.SharedMemory(size=x_reminder_nbytes)
                shm_Y_R = smm.SharedMemory(size=y_reminder_nbytes)
                RX = np.ndarray(x_reminder_shape, dtype=float_precision, buffer=shm_X_R.buf)
                RY = np.ndarray(y_reminder_shape, dtype=float_precision, buffer=shm_Y_R.buf)
                shared_data = (AX, AY, BX, BY, RX, RY)
            else:
                shared_data = (AX, AY, BX, BY)

            data_prep_process = multiprocessing.Process(target=self.data_prep, args=(idList_train_arr, idList_val_arr, reminder_buffer_size, img_size, num_class, scale, buffer_size, iteration, writer_info_q, reader_info_q, shared_data,))
            training_process = multiprocessing.Process(target=self.training, args=(writer_info_q, reader_info_q, shared_data, buffer_num, num_class, batch_size, epoch, reminder_buffer_size,save_path,))
            
            start = time()
            data_prep_process.start()
            training_process.start()

            data_prep_process.join()
            training_process.join()
            end = time()
            
            print('All time in processing: ', end- start)