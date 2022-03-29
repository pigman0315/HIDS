import os
import numpy as np
    
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
        
def analyze(dir_path,train_ratio):
    train,test_normal,test_attack = get_data(dir_path,train_ratio)
    train_dict = get_syscall_type(train)
    test_normal_dict = get_syscall_type(test_normal)
    test_attack_dict = get_syscall_type(test_attack)
    

if __name__ == '__main__': 
    DIR_PATH = '../../LID-DS/CVE-2012-2122'
    TRAIN_RATIO = 0.2
    analyze(DIR_PATH,TRAIN_RATIO)