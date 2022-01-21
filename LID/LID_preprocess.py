import os
import sys
from datetime import datetime
import numpy as np

class Preprocess:
    def __init__(self,seq_len=20,train_ratio=0.2,save_file_intvl=20):
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.save_file_intvl = save_file_intvl
        
    def remove_npy(self,dir_path):
        files = os.listdir(dir_path)
        for file in files:
            if(file[-4:] == '.npy'):
                print('remove {}'.format(file))
                os.remove(os.path.join(dir_path,file))
    
    def process_data(self,dir_path):
        # read syscall list
        syscall_num_map = {}
        f = open('./syscall_list.txt')
        next(f) # skip header
        for line in f.readlines():
            tokens = line[:-1].split(',')
            syscall_num_map[tokens[1]] = tokens[0]
        ######################## testing SYSCALL EMBEDDING ########################################
        # read syscall_vec
        syscall_embed = np.load('./syscall_embed_16.npz')['vec']
        #syscall_embed = np.eye(334,dtype=int)
        #syscall_embed = [i/334.0 for i in range(334)]
        ###########################################################################################

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
        valid_cnt = 0
        test_normal_cnt = 0
        for file_cnt, file_info in enumerate(normal_file_info_list):
            f = open(os.path.join(dir_path,file_info[1]))
            syscall_map = {} # map CPU# to syscall_list
            # syscall_map[0] = [] # testing
            for line in f.readlines():
                features = line.split(' ')
                if(features[6] == '>' and (features[7] in syscall_num_map.keys())): # 6: direction, 7: syscall name
                    # if(features[2] not in syscall_map.keys()): # 2: CPU#
                    #     syscall_map[features[2]] = []
                    # syscall_map[features[2]].append(syscall_num_map[features[7]])
                    if(features[5] not in syscall_map.keys()): # 2: CPU#
                        syscall_map[features[5]] = []
                    syscall_map[features[5]].append(syscall_num_map[features[7]])
                    #syscall_map[0].append(syscall_num_map[features[7]])
            f.close()
            for key in syscall_map.keys():
                #print(key,len(syscall_map[key]))
                for i in range(len(syscall_map[key])-self.seq_len+1):
                    #normal_syscall_seq.append([int(syscall)/334.0 for syscall in syscall_map[key][i:i+self.seq_len]])
                    normal_syscall_seq.append([syscall_embed[int(syscall)] for syscall in syscall_map[key][i:i+self.seq_len]])
            
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
                normal_syscall_seq = np.array(normal_syscall_seq)
                np.save(os.path.join(dir_path,'test_normal_'+str(test_normal_cnt)),normal_syscall_seq)
                print('{} shape = {}'.format('test_normal_'+str(test_normal_cnt),normal_syscall_seq.shape))
                test_normal_cnt += 1
                normal_syscall_seq = [] # clear normal_syscall_seq
        # get attack data
        attack_syscall_seq = []
        attack_cnt = 0
        for file_cnt,file_info in enumerate(attack_file_info_list):
            f = open(os.path.join(dir_path,file_info[1]))
            start_time = datetime.strptime(f.readline().split(' ')[1][:-3],"%H:%M:%S.%f")
            exploit_time = float(file_info[5])
            syscall_map = {} # map CPU# to syscall_list
            # syscall_map[0] = []
            for line in f.readlines():
                features = line.split(' ')
                cur_time = datetime.strptime(features[1][:-3],"%H:%M:%S.%f")
                time_delta = (cur_time-start_time).total_seconds()
                #if(features[6] == '>' and time_delta > exploit_time and features[7] in syscall_num_map.keys()): # 6: direction, 7: syscall name
                if(features[6] == '>' and features[7] in syscall_num_map.keys()): # 6: direction, 7: syscall name
                    # if(features[2] not in syscall_map.keys()): # 2: CPU#
                    #     syscall_map[features[2]] = []
                    # syscall_map[features[2]].append(syscall_num_map[features[7]])
                    if(features[5] not in syscall_map.keys()): # 2: CPU#
                        syscall_map[features[5]] = []
                    syscall_map[features[5]].append(syscall_num_map[features[7]])     
                    #syscall_map[0].append(syscall_num_map[features[7]]) 
            for key in syscall_map.keys():
                for i in range(len(syscall_map[key])-self.seq_len+1):
                    attack_syscall_seq.append([syscall_embed[int(syscall)] for syscall in syscall_map[key][i:i+self.seq_len]])
                    # attack_syscall_seq.append([int(syscall)/334.0 for syscall in syscall_map[key][i:i+self.seq_len]])
            
            # save each file's syscall
            if(len(attack_syscall_seq) > self.seq_len*20):
                attack_syscall_seq = np.array(attack_syscall_seq) # list to np.array
                np.save(os.path.join(dir_path,'test_attack_'+str(attack_cnt)),attack_syscall_seq) # save np.array
                print('{} shape = {}, {}'.format('test_attack_'+str(attack_cnt),attack_syscall_seq.shape,file_info[1]))
                attack_cnt += 1
                attack_syscall_seq = []
            f.close()
        # print file count of each category
        print("Train_cnt: {},  Test_normal_cnt: {}, Test_attack_cnt: {}".format(train_cnt,test_normal_cnt,attack_cnt))