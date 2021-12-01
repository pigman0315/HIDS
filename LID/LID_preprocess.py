import os
import sys

class Preprocess:
    def __init__(self,dir_path,seq_len=20):
        self.dir_path = dir_path
        self.seq_len = seq_len

    def filtering_and_abstraction(self,syscall_list):
        return

    def get_syscall_sequence_list(self,dir_path):
        syscall_seq_list = []
        files = os.listdir(dir_path)
        for file_name in files:
            f = open(os.path.join(dir_path,file_name))
            syscall_list = f.readline().split(' ')
            self.filtering_and_abstraction(syscall_list)
            for i in range(len(syscall_list)-self.seq_len):
                syscall_seq_list.append([float(syscall)/self.total_syscall_num for syscall in syscall_list[i:i+self.seq_len]])
        return syscall_seq_list
    
    def read_dir(self):
        # skip event type list
        skip_envtype_list = ['switch','procexit']

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
        f = open(os.path.join(self.dir_path,"runs.csv"))
        for line in f.readlines():
            file_info = line[:-1].split(', ')
            file_info[1] = file_info[1] + '.txt'
            if(file_info[2] == 'False'):
                normal_file_info_list.append(file_info)
            elif(file_info[2] == 'True'):
                attack_file_info_list.append(file_info)
        f.close()

        # get training data
        train_syscall_seq = []
        for file_info in normal_file_info_list:
            f = open(os.path.join(self.dir_path,file_info[1]))
            all_syscall_list = []
            for line in f.readlines():
                features = line.split(' ')
                if(features[6] == '>' and features[7] not in skip_envtype_list):
                    all_syscall_list.append(features[7])
            for i in range(len(all_syscall_list)-self.seq_len+1):
                train_syscall_seq.append([syscall_num_map[syscall] for syscall in all_syscall_list[i:i+self.seq_len]])
        print(len(train_syscall_seq))
        print(train_syscall_seq[0])
                

        # get validation data

        # get attack data