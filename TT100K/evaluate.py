from func import MyModel
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import os,shutil
import time
from matplotlib import pyplot as plt
import tensorflow as tf

ttcs = {}
btcs = {}
for rt in range(10):
    ttcs[rt] = pd.read_csv('/mnt/TTChina/result_0324/result_t/timecost_%d.csv' % rt)
    btcs[rt] = pd.read_csv('/mnt/TTChina/result_B0324/result_b/timecost_%d.csv' % rt)
print('OK',len(ttcs[0]),len(btcs[0]))

def mymeasure(_pred, _Y):
    t0 = time.time()
    # 效果评价
    acc = 0
    tp = 0 # 预测为1成功个数
    fp = 0 # 预测为1失败个数
    pp = 0 # 预测为1个数
     
    pos_idx = np.where(_Y==1)[0]
    _s = _pred.numpy()[pos_idx].min() - 0.01
    
    _pred01 = tf.cast(_pred>=_s, dtype=tf.int8)
    tp = np.sum(_pred01 * _Y )
    tn = np.sum((1-_pred01) * (1-_Y))
    fp = np.sum(_pred01 * (1-_Y))
    pp = np.sum(_pred01)
    acc = tp + tn

    # 精确率
    acc = acc / len(_Y)
    # 准确率
    tpr = tp / pp
    # 精确率
    fpr = fp / pp
    # auc
    auc = roc_auc_score(_Y,_pred)
    t1 = time.time()
    #print(t1-t0)
    #print(acc, auc, tpr, fpr, tp, fp)
    return acc, auc, tpr, fpr, tp, fp

        

def mymodelmeasure(model):
    # 测试模型
    res = []
    t0 = time.time()
    for _id in range(2834): #2834
        _X = np.load('/home/work/test/X/%d.npy' % _id).reshape([-1,512])
        _Y = np.load('/home/work/test/Y/%d.npy' % _id).reshape([-1,1])
        if _Y.sum() > 0:
            # 评价结果
            _res = mymeasure(model(_X), _Y)
            #print(_id, _res)
            res.append(_res)
        if _id % 500 == 0:
            print(_id, len(res), '%.2fs' % (time.time()-t0))
    res = pd.DataFrame(res, columns=['acc', 'auc', 'tpr', 'fpr', 'tp', 'fp'])
    return res

def add_measure(rt, _tc, op):
    # 测试方法
    new_res = []
    for i in range(len(_tc)):
        t0 = time.time()
        _line = _tc.iloc[i]
        new_line = list(_line)
        if op == 'btc':
            model_path = '/home/mnt/TTChina/result_B0324/model_b/%d=%d-%d.h5' % (rt, _line['step'], _line['niter'])
        elif op == 'ttc':
            model_path = '/home/mnt/TTChina/result_0324/model_t/%d=%d.h5' % (rt,_line['epoch'])
        
        # 读取模型
        model = MyModel()   
        model.load_weights(model_path)
        print('%d %s-%d Start' % (rt, op, i))
        
        # 测试模型
        res = mymodelmeasure(model)
        new_line = new_line + [res['acc'].mean(), res['auc'].mean(),
                               res['tpr'].mean(), res['tp'].mean(),
                               res['fpr'].median(), res['fp'].median(),
                              res['fpr'].mean(), res['fp'].mean()]
        new_res.append(new_line)
        t1 = time.time()
        if op == 'btc':
            print('%d=%d-%d. %.2fs. %.2f' % (rt,_line['step'], _line['niter'], t1-t0, res['fp'].median()))
        elif op == 'ttc':
            print('%d=%d. %.2fs. %.2f' % (rt,_line['epoch'], t1-t0, res['fp'].median()))
    new_res = pd.DataFrame(new_res, columns=list(_tc.columns)+
                          ['acc', 'auc', 'tpr', 'tp', 'fpr', 'fp', 'fpr_m', 'fp_m'])
    new_res.to_csv('stat/%d_%s_stat.csv' % (rt, op), index=False)
    return new_res


import pandas as pd
from matplotlib import pyplot as plt

RTN = 10

btcs = {}
ttcs = {}
for rt in range(RTN):
    btc = pd.read_csv('stat/%d_%s_stat.csv' % (rt, 'btc'))
    ttc = pd.read_csv('stat/%d_%s_stat.csv' % (rt, 'ttc'))
    
    
    # 计算累计训练时间
    _s = .0
    _tc = []
    for i in range(len(btc)):
        _line = btc.iloc[i]
        _s += _line['tc_buffer'] + _line['tc_train']
        _tc.append(_s)
    btc['tc'] = _tc
    
    # 计算累计训练时间
    _s = .0
    _tc = []
    for i in range(len(ttc)):
        _line = ttc.iloc[i]
        _s += _line['tc_train']
        _tc.append(_s)
    ttc['tc'] = _tc
    
    btcs[rt] = btc
    ttcs[rt] = ttc

def summary_tar(tar, _tcs, init_value):
    # 记录平均的时间、指标
    _tc = []
    _n = len(_tcs[0])
    for i in range(_n):
        _s = .0
        for rt in range(RTN):
            _s += _tcs[rt]['tc'].iloc[i]
        _tc.append(_s / 10)

    _tar = []
    for i in range(_n):
        _s = .0
        for rt in range(RTN):
            _s += _tcs[rt][tar].iloc[i]
        _tar.append(_s/10)
    return [0] + _tc, [init_value] + _tar

def summary_init_tar():
    # 记录平均的初始
    btcd = []
    ttcd = []
    for rt in range(RTN):
        btcd.append( pd.read_csv('stat/%d_%s_stat_init.csv' % (rt, 'btc')) )
        ttcd.append( pd.read_csv('stat/%d_%s_stat_init.csv' % (rt, 'ttc')) )
    btcd = pd.concat(btcd, axis=0)
    ttcd = pd.concat(ttcd, axis=0)
    return btcd, ttcd
    
b_init, t_init = summary_init_tar()

btc_tc, btc_tar = summary_tar('fp', btcs, b_init['fp'].mean())
ttc_tc, ttc_tar = summary_tar('fp', ttcs, t_init['fp'].mean())

plt.plot(btc_tc, btc_tar, label='BMGD')
plt.plot(ttc_tc, ttc_tar, label='Default')
plt.grid(linestyle=':')
plt.ylabel(r'$\widebar{FP}$')
plt.legend()
plt.xlabel('Time cost (second)')

btc_tc, btc_tar = summary_tar('auc', btcs, b_init['auc'].mean())
ttc_tc, ttc_tar = summary_tar('auc', ttcs, t_init['auc'].mean())

plt.plot(btc_tc, btc_tar, label='BMGD')
plt.plot(ttc_tc, ttc_tar, label='Default')
plt.grid(linestyle=':')
plt.ylabel(r'$\widebar{AUC}$')
plt.legend()
plt.xlabel('Time cost (second)')
