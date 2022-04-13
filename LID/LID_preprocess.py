import os
import sys
from datetime import datetime
import numpy as np

class Preprocess:
    def __init__(self,seq_len=20,train_ratio=0.2,save_file_intvl=20):
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.save_file_intvl = save_file_intvl
        self.target_syscall_w = ['write','writev','pwrite','pwritev','pwritev2']
        self.target_syscall_r = ['read','readv','pread','preadv','preadv2']
        
    def remove_npy(self,dir_path):
        files = os.listdir(dir_path)
        for file in files:
            if(file[-4:] == '.npy'):
                print('remove {}'.format(file))
                os.remove(os.path.join(dir_path,file))

    def get_syscall_key(self,features):
        # key consists of syscall_name + return value(file size) + file_name(may be ip or file)
        # only consider return value & file name of syscall in target syscall list
        # other syscall only consider its syscall name
        key = ""
        syscall_name = features[7]
        if(syscall_name in self.target_syscall_w):
            key += 'write_'
        elif(syscall_name in self.target_syscall_r):
            key += 'read_'
        else:
            key += syscall_name
        if(syscall_name in self.target_syscall_w or syscall_name in self.target_syscall_r):
            # get ret_value
            ret_value = features[9].split('=')[1]
            key += ret_value + '_'
            # get file_name
            try:
                fd_content = features[8].split('(')[1].split(')')[0]
                if('<4t>' in fd_content):
                    file_name = fd_content.split('>')[2]    
                else:
                    file_name = fd_content.split('>')[1]
                    if('tmp' in file_name):
                        file_name = 'tmp'
                    elif('pipe' in file_name):
                        file_name = 'pipe'
            except:
                file_name = ''
            key += file_name
        return key
        

    def process_data(self,dir_path):
        # read syscall list
        syscall_name_set = set()
        f = open('./syscall_list.txt')
        next(f) # skip header
        for line in f.readlines():
            syscall_name = line[:-1].split(',')[1]
            syscall_name_set.add(syscall_name)

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
        normal_syscall_seq = []
        train_cnt = 0
        test_normal_cnt = 0
        map_syscallArg_to_idx = {} # map 'syscall_name+ret_value+file_name' to a unique idx
        for file_cnt, file_info in enumerate(normal_file_info_list):
            #print('map size: {}'.format(len(map_syscallArg_to_idx)))
            f = open(os.path.join(dir_path,file_info[1]))
            map_threadN_to_syscall = {} # put all syscalls with same thread# into a list
            for line in f.readlines():
                features = line.split(' ')
                syscall_name = features[7]
                syscall_direction = features[6]
                thread_num = features[5]
                if(syscall_direction == '>' and syscall_name in syscall_name_set):
                    # get index in syscall_map with key 'syscall+ret_val+file'
                    key = self.get_syscall_key(features)
                    syscall_idx = -100000 
                    if(file_cnt < int(len(normal_file_info_list)*self.train_ratio)): # training file
                        if(key not in map_syscallArg_to_idx.keys()):
                            #print(key)
                            map_syscallArg_to_idx[key] = len(map_syscallArg_to_idx)
                        syscall_idx = map_syscallArg_to_idx[key]
                    else:                                                            # test_normal file
                        if(key not in map_syscallArg_to_idx.keys()):
                            if(syscall_name in self.target_syscall_w):
                                syscall_idx = len(map_syscallArg_to_idx) # represent unknown write
                            elif(syscall_name in self.target_syscall_r):
                                syscall_idx = len(map_syscallArg_to_idx)+1 # represent unknown read
                            else:
                                syscall_idx = len(map_syscallArg_to_idx)+2 # represent unknown syscall
                        else:
                            syscall_idx = map_syscallArg_to_idx[key]
                        
                    # threadID major
                    if(thread_num not in map_threadN_to_syscall.keys()):
                        map_threadN_to_syscall[thread_num] = []
                    map_threadN_to_syscall[thread_num].append(syscall_idx)  
            f.close()
            for key in map_threadN_to_syscall.keys():
                for i in range(len(map_threadN_to_syscall[key])-self.seq_len+1):
                    normal_syscall_seq.append(map_threadN_to_syscall[key][i:i+self.seq_len])
            
            # training data, save data for every "self.save_file_intvl"
            if(file_cnt < int(len(normal_file_info_list)*self.train_ratio)):
                if((file_cnt+1) % self.save_file_intvl == 0 
                    or file_cnt == int(len(normal_file_info_list)*self.train_ratio)-1
                ):
                    normal_syscall_seq = np.array(normal_syscall_seq)
                    np.save(os.path.join(dir_path,'train_'+str(train_cnt)),normal_syscall_seq)
                    print('{} shape = {}'.format('train_'+str(train_cnt),normal_syscall_seq.shape))
                    train_cnt += 1
                    normal_syscall_seq = [] # clear normal_syscall_seq
                else:
                    continue
            else:
                if(len(normal_syscall_seq) > self.seq_len*1):
                    normal_syscall_seq = np.array(normal_syscall_seq)
                    np.save(os.path.join(dir_path,'test_normal_'+str(test_normal_cnt)),normal_syscall_seq)
                    print('{} shape = {}, {}'.format('test_normal_'+str(test_normal_cnt),normal_syscall_seq.shape,file_info[1]))
                    test_normal_cnt += 1
                normal_syscall_seq = [] # clear normal_syscall_seq
        
        # get attack data
        attack_syscall_seq = []
        attack_cnt = 0
        for file_cnt, file_info in enumerate(attack_file_info_list):
            f = open(os.path.join(dir_path,file_info[1]))
            map_threadN_to_syscall = {} # put all syscalls with same thread# into a list
            for line in f.readlines():
                features = line.split(' ')
                syscall_name = features[7]
                syscall_direction = features[6]
                thread_num = features[5]
                if(syscall_direction == '>' and syscall_name in syscall_name_set):
                    # get index in syscall_map with key 'syscall+ret_val+file'
                    key = self.get_syscall_key(features)
                    syscall_idx = -100000 
                    if(key not in map_syscallArg_to_idx.keys()):
                        if(syscall_name in self.target_syscall_w):
                            syscall_idx = len(map_syscallArg_to_idx) # represent unknown write
                        elif(syscall_name in self.target_syscall_r):
                            syscall_idx = len(map_syscallArg_to_idx)+1 # represent unknown read
                        else:
                            syscall_idx = len(map_syscallArg_to_idx)+2 # represent unknown syscall
                    else:
                        syscall_idx = map_syscallArg_to_idx[key]
                        
                    # threadID major
                    if(thread_num not in map_threadN_to_syscall.keys()):
                        map_threadN_to_syscall[thread_num] = []
                    map_threadN_to_syscall[thread_num].append(syscall_idx)  
            f.close()
            for key in map_threadN_to_syscall.keys():
                for i in range(len(map_threadN_to_syscall[key])-self.seq_len+1):
                    attack_syscall_seq.append(map_threadN_to_syscall[key][i:i+self.seq_len])
            # save each file's syscall
            attack_syscall_seq = np.array(attack_syscall_seq) # list to np.array
            np.save(os.path.join(dir_path,'test_attack_'+str(attack_cnt)),attack_syscall_seq) # save np.array
            print('{} shape = {}, {}'.format('test_attack_'+str(attack_cnt),attack_syscall_seq.shape,file_info[1]))
            attack_cnt += 1
            attack_syscall_seq = []

        # print file count of each category
        print("Train_cnt: {},  Test_normal_cnt: {}, Test_attack_cnt: {}".format(train_cnt,test_normal_cnt,attack_cnt))

        # save map_syscallArg_to_idx size
        map_size = np.array([len(map_syscallArg_to_idx)+3]) # 3: unknown read, unknown write, unknown syscall
        print('Map size = {}'.format(map_size[0]))
        np.save(os.path.join(dir_path,'map_size'),map_size)