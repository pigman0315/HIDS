import os
import sys
from datetime import datetime
import numpy as np

class Preprocess:
    def __init__(self,seq_len=20,train_ratio=0.2):
        self.seq_len = seq_len
        self.train_ratio = train_ratio

    def filtering_and_abstraction(self,syscall_list):
        return
    
    def process_data(self,dir_path):
        # read syscall list
        syscall_num_map = {}
        f = open('./syscall_list.txt')
        next(f) # skip header
        for line in f.readlines():
            tokens = line[:-1].split('\t')
            syscall_num_map[tokens[1]] = tokens[0]

        # read run.csv
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
        normal_syscall_seq = []
        for file_info in normal_file_info_list:
            f = open(os.path.join(dir_path,file_info[1]))
            all_syscall_list = []
            start_time = datetime.strptime(f.readline().split(' ')[1][:-3],"%H:%M:%S.%f")
            warmup_time = float(file_info[3])
            for line in f.readlines():
                features = line.split(' ')
                cur_time = datetime.strptime(features[1][:-3],"%H:%M:%S.%f")
                time_delta = (cur_time-start_time).total_seconds()
                if(features[6] == '>' and time_delta > warmup_time and features[7] in syscall_num_map.keys()):
                    all_syscall_list.append(features[7])
            f.close()
            for i in range(len(all_syscall_list)-self.seq_len+1):
                 normal_syscall_seq.append([float(syscall_num_map[syscall])/len(syscall_num_map) for syscall in all_syscall_list[i:i+self.seq_len]])
        train_syscall_seq = normal_syscall_seq[:int(len(normal_syscall_seq)*self.train_ratio)]
        valid_syscall_seq = normal_syscall_seq[int(len(normal_syscall_seq)*self.train_ratio):]

        train_syscall_seq = np.array(train_syscall_seq) # list to np.array
        valid_syscall_seq = np.array(valid_syscall_seq) # list to np.array
        np.save(os.path.join(dir_path,'train'),train_syscall_seq) # save np.array
        np.save(os.path.join(dir_path,'valid'),valid_syscall_seq) # save np.array
        print(train_syscall_seq.shape)
        print(valid_syscall_seq.shape)

        # get attack data
        attack_syscall_seq = []
        for file_info in attack_file_info_list:
            f = open(os.path.join(dir_path,file_info[1]))
            all_syscall_list = []
            start_time = datetime.strptime(f.readline().split(' ')[1][:-3],"%H:%M:%S.%f")
            exploit_time = float(file_info[5])
            for line in f.readlines():
                features = line.split(' ')
                cur_time = datetime.strptime(features[1][:-3],"%H:%M:%S.%f")
                time_delta = (cur_time-start_time).total_seconds()
                if(features[6] == '>' and time_delta > exploit_time and features[7] not in skip_envtype_list):
                    all_syscall_list.append(features[7])
            f.close()
            for i in range(len(all_syscall_list)-self.seq_len+1):
                attack_syscall_seq.append([float(syscall_num_map[syscall])/len(syscall_num_map) for syscall in all_syscall_list[i:i+self.seq_len]])
            
        attack_syscall_seq = np.array(attack_syscall_seq) # list to np.array
        np.save(os.path.join(dir_path,'attack'),attack_syscall_seq) # save np.array
        print(attack_syscall_seq.shape)