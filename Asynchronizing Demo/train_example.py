import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 引入自定义的网络模型构建方法
from netmodels import build_AlexNet, build_ResNet

# 引入BMGD
import BMGD 



if __name__ == '__main__':
    # 初始化一个BMGD训练对象
    worker = BMGD.BMGD_worker(
        model_build=build_AlexNet, # 模型定义方法
        model_build_info={'im_size':128, 'num_class':2}, # 模型定义需要的超参数
        loss='categorical_crossentropy', # 损失函数
        epoch=10, # buffer内训练的 epoch数
        iteration=10, # buffer迭代次数
        batch_size=150, # batch size
        buffer_size=15000, # buffer size
        save_path='/home/result_0809/', # 结果保存位置
        img_size=128, # 输入图像大小
        channel_size=3, # 输入通道大小
        num_class=2, # 分类类别
        scale=255, # 像素范围
        train_file_list='filepath.csv', # 训练数据路径文件
        val_file_list='filepath.csv' # 验证数据路径文件
    )
  
    # 可选项（默认false关闭）
    worker.save_all_checkpoints = True
    # 是否图像增强
    worker.augment = False
    # 是否记录时间
    worker.record = True
        
    # 启动BMGD训练
    worker.main()

   