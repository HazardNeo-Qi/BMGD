import os
import tensorflow as tf
import config as c
import time
import sys
from tqdm import tqdm
from tensorflow.keras import optimizers
from utils.data_utils import train_iterator, train_iterator_buffer, read_a_buffer
from utils.data_utils import test_iterator
from utils.eval_utils import cross_entropy_batch, correct_num_batch, l2_loss
from model.ResNet import ResNet
from model.ResNet_v2 import ResNet_v2
from test import test_bmgd
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 可见GPU设置
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():    
 
    class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
        def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
            self.warm_up_step = warm_up_step
            super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                        decay_steps=decay_steps,
                                                        alpha=alpha,
                                                        name=name)

        @tf.function
        def __call__(self, step):
            if step <= self.warm_up_step:
                return step / self.warm_up_step * self.initial_learning_rate
            else:
                return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)

    @tf.function
    def train_step(model, images, labels, optimizer):
        with tf.GradientTape() as tape:
            prediction = model(images, training=True)
            ce = cross_entropy_batch(labels, prediction, label_smoothing=c.label_smoothing)           
            l2 = l2_loss(model)
            loss = ce + l2
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return ce, prediction

    @tf.function # 并没有试过去掉这个注解会造成什么后果
    def distributed_train_step(model, images, labels, optimizer):
        per_replica_losses = strategy.run(
            train_step, args=(model, images, labels, optimizer)
        )
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def train(epoch_num, model, optimizer):
        mylog('epoch:{}'.format(epoch_num))
        
        # buffer 次数
        buffer_num = int( c.train_num / c.buffer_size ) #+ 1
        
        for _b in range(buffer_num):
            # 获取一个大buffer
            t0 = time.time()
            # 读取一片数据
            buffer_images, buffer_labels = read_a_buffer(c.buffer_size*_b, c.buffer_size*(_b+1))
            mylog('epoch-batch: %d - %d. buffer loading: %.2fs' % (epoch_num, _b, time.time()-t0))
            
            # 对buffer产生数据生成器
            tmp_data_generator = train_iterator_buffer(buffer_images, buffer_labels)
    
            
            for buffer_epoch in range(10):
                # 单个buffer重复利用！
                mylog('epoch-batch-ID: %d - %d - %d' % (epoch_num, _b, buffer_epoch))
                
                
                _ids = int(c.buffer_size / c.batch_size) + 1
                # 内存保存buffer数据
                t0 = time.time()
                buffer_bag = []
                for _id in tqdm(range(_ids)):
                    images, labels = tmp_data_generator.next()
                    buffer_bag.append( [images, labels] )
                mylog('buffer_once_loading {:.4f}'.format(time.time()-t0))   
            
                
                sum_ce = 0
                sum_correct_num = 0

                t0 = time.time()
                # 用小batch遍历这个大buffer
                for _id in tqdm(range(_ids)):
                    # 从buffer数据直接拿mini batch，不再读原始数据
                    #images, labels = tmp_data_generator.next()
                    [images, labels] = buffer_bag[_id]
                    ce, prediction = distributed_train_step(model, images, labels, optimizer)
                    correct_num = correct_num_batch(labels, prediction)
                    sum_ce += ce * c.batch_size
                    sum_correct_num += correct_num                

                    print('ce: {:.4f}, accuracy: {:.4f}, l2 loss: {:.4f}'.format(ce, correct_num / c.batch_size, l2_loss(model)))

                mylog('train: cross entropy loss: {:.4f}, l2 loss: {:.4f}, accuracy: {:.4f}, time: {:.4f}'.format(
                    sum_ce / c.train_num,l2_loss(model),sum_correct_num / c.train_num, time.time()-t0))

                # 每个buffer记录一下测试集表现
                test_bmgd(model)

                
def mylog(info):
    with open(c.log_file_buffer, 'a') as f:    
        f.write(info+'\n')
        
        
if __name__ == '__main__':
    # gpu config
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    
    print(c.initial_learning_rate)


    with strategy.scope(): 
        # load data
        test_data_iterator = test_iterator()

        # get model
        # model = ResNet(50)
        model = ResNet_v2(50)

        # show
        model.build(input_shape=(None,) + c.input_shape)
        model.summary()
        print('initial l2 loss:{:.4f}'.format(l2_loss(model)))
       
        
        # load pretrain
        if c.load_weight_file is not None:
            model.load_weights(c.load_weight_file)
            print('pretrain weight l2 loss:{:.4f}'.format(l2_loss(model)))

        # train
        learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=c.initial_learning_rate,
                                                        decay_steps=c.epoch_num * c.iterations_per_epoch - c.warm_iterations,
                                                        alpha=c.minimum_learning_rate,
                                                        warm_up_step=c.warm_iterations)
        optimizer = optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)
        for epoch_num in range(c.epoch_num):
            train(epoch_num, model, optimizer)
                

            model.save_weights(c.save_weight_file, save_format='h5')

            # save intermediate results
            if epoch_num % 5 == 4:
                os.system('cp {} {}_epoch_{}.h5'.format(c.save_weight_file, c.save_weight_file.split('.')[0], epoch_num))
