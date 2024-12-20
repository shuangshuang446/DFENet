import scipy.io as scio
import numpy as np
import os
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *


def UNIMIB(dataset_dir='./UniMiB-SHAR/data', SPLIT_RATE=(8, 2), VALIDATION_SUBJECTS={}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):

    if VALIDATION_SUBJECTS:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (VALIDATION_SUBJECTS))
        for each in VALIDATION_SUBJECTS:
            assert each in set([*range(1, 31)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))

    dataset_dir = './data/UniMiB-SHAR/data'


    dir = dataset_dir
    data = scio.loadmat(os.path.join(dir, 'acc_data.mat'))['acc_data']
    label = scio.loadmat(os.path.join(dir, 'acc_labels.mat'))['acc_labels']

    xtrain, ytrain, xtest, ytest = [], [], [], []

    print('Loading subject data')
    for subject_id in range(1, 31):

        print('     current subject: 【%d】'%(subject_id), end='')
        print('   ----   Validation Data' if subject_id in VALIDATION_SUBJECTS else '')

        for label_id in range(17):
            mask = np.logical_and(label[:, 0] == label_id+1, label[:, 1] == subject_id)
            cur_data = data[mask].tolist()
            

            if VALIDATION_SUBJECTS: # 留一法
                if subject_id not in VALIDATION_SUBJECTS:
                    xtrain += cur_data
                    ytrain += [label_id] * len(cur_data)
                else: # 验证集
                    xtest += cur_data
                    ytest += [label_id] * len(cur_data)
            else: # 平均法
                trainlen = int(len(cur_data) * SPLIT_RATE[0] / sum(SPLIT_RATE))
                testlen = len(cur_data) - trainlen
                xtrain += cur_data[:trainlen]
                xtest += cur_data[trainlen:]
                ytrain += [label_id] * trainlen
                ytest += [label_id] * testlen

    xtrain, ytrain, xtest, ytest = np.array(xtrain), np.array(ytrain), np.array(xtest), np.array(ytest)


    xtrain = xtrain.reshape(xtrain.shape[0], 3, 151).transpose(0, 2, 1)
    xtest = xtest.reshape(xtest.shape[0], 3, 151).transpose(0, 2, 1)

    if Z_SCORE:
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)

    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH:
        save_npy_data(
            dataset_name='UniMiB_SHAR',
            root_dir=SAVE_PATH,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )
            
    return xtrain, xtest, ytrain, ytest


if __name__ == '__main__':
    UNIMIB()
