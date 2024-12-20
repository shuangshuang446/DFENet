import os
import numpy as np
import pandas as pd
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *


def PAMAP(dataset_dir='./PAMAP2_Dataset/Protocol', WINDOW_SIZE=171, OVERLAP_RATE=0.5, SPLIT_RATE=(8, 2), VALIDATION_SUBJECTS={105}, Z_SCORE=True, SAVE_PATH=os.path.abspath('../../HAR-datasets')):

    if VALIDATION_SUBJECTS:
        print('\n---------- 采用【留一法】分割验证集，选取的subject为:%s ----------\n' % (VALIDATION_SUBJECTS))
        for each in VALIDATION_SUBJECTS:
            assert each in set([*range(101, 110)])
    else:
        print('\n---------- 采用【平均法】分割验证集，训练集与验证集样本数比为:%s ----------\n' % (str(SPLIT_RATE)))


    download_dataset(
        dataset_name='PAMAP2',
        file_url='http://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip',
        dataset_dir=dataset_dir
    )

    xtrain, xtest, ytrain, ytest = [], [], [], [] # train-test-data, 用于存放最终数据
    category_dict = dict(zip([*range(12)], [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])) #12分类所对应的实际label，对应readme.pdf

    dir = dataset_dir
    filelist = os.listdir(dir)
    os.chdir(dir)
    print('Loading subject data')
    for file in filelist:
        
        subject_id = int(file.split('.')[0][-3:])
        print('     current subject: 【%d】'%(subject_id), end='')
        print('   ----   Validation Data' if subject_id in VALIDATION_SUBJECTS else '')

        content = pd.read_csv(file, sep=' ', usecols=[1]+[*range(4,16)]+[*range(21,33)]+[*range(38,50)])
        content = content.interpolate(method='linear', limit_direction='forward', axis=0).to_numpy()
        

        data = content[::3, 1:]
        label = content[::3, 0]

        data = data[label!=0]
        label = label[label!=0]

        for label_id in range(12):
            true_label = category_dict[label_id]
            cur_data = sliding_window(array=data[label==true_label], windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)


            if VALIDATION_SUBJECTS:
                if subject_id not in VALIDATION_SUBJECTS:
                    xtrain += cur_data
                    ytrain += [label_id] * len(cur_data)
                else:
                    xtest += cur_data
                    ytest += [label_id] * len(cur_data)
            else: # 平均法
                trainlen = int(len(cur_data) * SPLIT_RATE[0] / sum(SPLIT_RATE))
                testlen = len(cur_data) - trainlen
                xtrain += cur_data[:trainlen]
                xtest += cur_data[trainlen:]
                ytrain += [label_id] * trainlen
                ytest += [label_id] * testlen

    os.chdir('../')

    xtrain = np.array(xtrain, dtype=np.float32)
    xtest = np.array(xtest, dtype=np.float32)
    ytrain = np.array(ytrain, np.int64)
    ytest = np.array(ytest, np.int64)

    if Z_SCORE:
        xtrain, xtest = z_score_standard(xtrain=xtrain, xtest=xtest)
    
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('xtrain shape: %s\nxtest shape: %s\nytrain shape: %s\nytest shape: %s'%(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape))

    if SAVE_PATH:
        save_npy_data(
            dataset_name='PAMAP2',
            root_dir=SAVE_PATH,
            xtrain=xtrain,
            xtest=xtest,
            ytrain=ytrain,
            ytest=ytest
        )
        
    return xtrain, xtest, ytrain, ytest


if __name__ == '__main__':
    PAMAP()
