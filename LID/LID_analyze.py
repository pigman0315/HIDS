import os
import numpy as np
import re
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
from LID_preprocess import Preprocess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader    
from LID_model import CAE

def get_data(dir_path,train_ratio):
    # read runs.csv
    normal_file_info_list = []
    attack_file_info_list = []
    f = open(os.path.join(dir_path,"runs.csv"))
    for line in f.readlines():
        file_info = line[:-1].split(', ')
        file_info[1] = file_info[1] + '.txt'
        if(file_info[2] == 'False'):
            normal_file_info_list.append(file_info)
        elif(file_info[2] == 'True'):
            attack_file_info_list.append(file_info)
    f.close()

    # get normal data
    train_syscall_list = []
    test_normal_syscall_list = []
    train_cnt = 0
    test_normal_cnt = 0
    
    for file_cnt, file_info in enumerate(normal_file_info_list):
        f = open(os.path.join(dir_path,file_info[1]))
        if(file_cnt < int(len(normal_file_info_list)*train_ratio)):
            train_cnt += 1
            print("train({})".format(train_cnt))
            for line in f.readlines():
                features = line.split(' ')
                if(features[6] == '>'):
                    train_syscall_list.append(features[7])
        else:
            test_normal_cnt += 1
            print("test_normal({})".format(test_normal_cnt))
            for line in f.readlines():
                features = line.split(' ')
                if(features[6] == '>'):
                    test_normal_syscall_list.append(features[7])

    # get attack data
    attack_syscall_list = []
    attack_cnt = 0
    for file_cnt,file_info in enumerate(attack_file_info_list):
        f = open(os.path.join(dir_path,file_info[1]))
        attack_cnt += 1
        print("test_attack({})".format(attack_cnt))
        for line in f.readlines():
            features = line.split(' ')
            if(features[6] == '>'):
                attack_syscall_list.append(features[7])

    # return data
    return train_syscall_list,test_normal_syscall_list,attack_syscall_list

def get_syscall_type(syscall_list):
    d = {}
    for syscall in syscall_list:
        if(syscall not in d.keys()):
            d[syscall] = 1
        else:
            d[syscall] += 1
    print(sorted(d.items(), key=lambda x:x[1]))
    print(len(d.keys()))
    print("---------------------------------")
    return d

def build_database(type,window_len):
    # get npy file
    file_list = os.listdir(DIR_PATH)
    find_pattern = re.compile(rf"{type}_[0-9]*\.npy") # $type_
    npy_list = find_pattern.findall(' '.join(file_list))
    
    # read data
    s = set()
    comb_cnt = 0
    for npy_file in npy_list:
        data = np.load(os.path.join(DIR_PATH,npy_file))
        comb_cnt += len(data)
        for seq in data:
            seq = [int(i*334) for i in seq]
            s.add(tuple(seq))
    print("{} combinations in {}".format(len(s),comb_cnt))
    return s

def analyze(dir_path,train_ratio,window_len):
    # train,test_normal,test_attack = get_data(dir_path,train_ratio)
    # train_dict = get_syscall_type(train)
    # test_normal_dict = get_syscall_type(test_normal)
    # test_attack_dict = get_syscall_type(test_attack)
    db_train = build_database('train',window_len)
    db_test_normal = build_database('test_normal',window_len)
    db_test_attack = build_database('test_attack',window_len)
    diff_train_normal = db_train - db_test_normal
    diff_train_attack = db_train - db_test_attack
    diff_attack_train = db_test_attack - db_train
    diff_normal_train = db_test_normal - db_train
    print("Train - Test_normal: {}\nTrain - Attack: {}\nAttack - Train: {}\nTest_normal - Train: {}".format(
        len(diff_train_normal),
        len(diff_train_attack),
        len(diff_attack_train),
        len(diff_normal_train)))                       

if __name__ == '__main__': 
    DIR_PATH = '../../LID-DS/CVE-2017-7529'
    TRAIN_RATIO = 0.2
    WINDOW_LEN = 20
    SAVE_FILE_INTVL = 50
    NEED_PREPROCESS = False
    
    if(NEED_PREPROCESS):
        prep = Preprocess(seq_len=WINDOW_LEN
            ,train_ratio=TRAIN_RATIO
            ,save_file_intvl=SAVE_FILE_INTVL
            )
        prep.remove_npy(DIR_PATH)
        prep.process_data(DIR_PATH)

    analyze(DIR_PATH,TRAIN_RATIO,WINDOW_LEN)