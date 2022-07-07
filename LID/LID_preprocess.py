import os
import sys
import collections
import numpy as np

class Preprocess:
    def __init__(self,seq_len=20,train_ratio=0.2,save_file_intvl=20,validation_ratio=0.2):
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.save_file_intvl = save_file_intvl
        self.validation_ratio = validation_ratio
        self.target_syscall_w = ['write','writev','pwrite','pwritev','pwritev2']
        self.target_syscall_r = ['read','readv','pread','preadv','preadv2',]
        
    def remove_npy(self,dir_path):
        files = os.listdir(dir_path)
        for file in files:
            if(file[-4:] == '.npy'):
                print('remove {}'.format(file))
                os.remove(os.path.join(dir_path,file))

    def get_syscall_key(self,features,nl_features=None):
        # key consists of syscall_name + return value(file size) + file_name(may be ip or file)
        # only consider return value & file name of syscall in target syscall list
        # other syscall only consider its syscall name
        key = ""
        syscall_name = features[7]
        if(syscall_name in self.target_syscall_w):
            key = 'write_'
        elif(syscall_name in self.target_syscall_r):
            key = 'read_'
        else:
            key = syscall_name
        if(syscall_name in self.target_syscall_w or syscall_name in self.target_syscall_r):
            # get ret_value
            ret_value = ''
            if(nl_features != None and nl_features[7] == syscall_name):
                ret_value = nl_features[8] # res    
            # get file_name
            try:
                fd_content = features[8].split('(')[1].split(')')[0]
                if('<4t>' in fd_content):
                    file_name = fd_content.split('>')[2]  
                    if('172.17' in file_name):
                        file_name = '172.17'
                else:
                    file_name = fd_content.split('>')[1]
                    if('tmp' in file_name):
                        file_name = 'tmp'
                        ret_value=''
                    elif('pipe' in file_name):
                        file_name = 'pipe'
                    elif('log' in file_name):
                        file_name = 'log'
                        ret_value = ''
                    elif('libzip' in file_name):
                        file_name = 'libzip'
                        ret_value = ''
                    elif('/dev/pts' in file_name):
                        file_name = '/dev/pts'
                    elif('/etc' in file_name):
                        file_name = '/etc'
                    elif(':' in file_name):
                        file_name = ''
                    elif('tomcat' in file_name):
                        file_name = 'tomcat'
                    elif('xml' in file_name):
                        file_name = 'xml'
                        ret_value = ''
                    elif('jdk' in file_name):
                        file_name = 'jdk'
                    elif(file_name == ''):
                        ret_value = ''
                    elif('sessions' in file_name):
                        file_name = 'sess'
                        ret_value = ''
                    elif('image' in file_name):
                        file_name = 'image'
            except:
                file_name = ''
                ret_value = ''
            key += file_name + '_'
            key += ret_value
        return key
        

    def process_data(self,dir_path):
        # read syscall list
        map_syscallName_to_syscallNum = {}
        f = open('./syscall_list.txt')
        for line in f.readlines():
            syscall_num = int(line[:-1].split(',')[0])
            syscall_name = line[:-1].split(',')[1]
            map_syscallName_to_syscallNum[syscall_name] = syscall_num

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
        valid_normal_cnt = 0
        test_normal_cnt = 0
        set_syscallArg = set() # set 'syscall_name+ret_value+file_name' as a unique key
        set_syscallArg.add('read__')
        set_syscallArg.add('write__')
        for file_cnt, file_info in enumerate(normal_file_info_list):
            f = open(os.path.join(dir_path,file_info[1]))
            map_threadN_to_syscall = collections.OrderedDict() # put all syscalls with same thread# into a list
            syscall_cnt = 0
            map_threadN_to_records = collections.OrderedDict()
            # categorize records by thread ID
            for line in f.readlines():
                features = line.split(' ')
                thread_num = features[5]
                if(thread_num not in map_threadN_to_records.keys()):
                    map_threadN_to_records[thread_num] = []
                map_threadN_to_records[thread_num].append(line)
            
            # convert syscall records to syscall number
            for tid in map_threadN_to_records.keys():
                map_threadN_to_syscall[tid] = []
                t_list = map_threadN_to_records[tid] # records list of thread_num
                for line_n in range(len(t_list)):
                    features = t_list[line_n].split(' ') 
                    syscall_direction = features[6]
                    syscall_name = features[7]
                    if(syscall_direction == '>' and syscall_name in map_syscallName_to_syscallNum.keys()):
                        # get key
                        syscall_cnt += 1
                        if(syscall_name in self.target_syscall_w or syscall_name in self.target_syscall_r):
                            try:
                                next_line_features = t_list[line_n+1].split(' ')
                                key = self.get_syscall_key(features,next_line_features)
                            except:
                                key = self.get_syscall_key(features)
                        else:
                            key = self.get_syscall_key(features)
                        # build normal key database
                        if(file_cnt < int(len(normal_file_info_list)*self.train_ratio)): # training file
                            if(key not in set_syscallArg):
                                print(key)
                                set_syscallArg.add(key)
                            syscall_idx = map_syscallName_to_syscallNum[syscall_name]
                        # check if key in database, and assign it a syscall index
                        else:                                                            # test_normal file
                            if(key not in set_syscallArg):
                                if(syscall_name in self.target_syscall_w):
                                    syscall_idx = len(map_syscallName_to_syscallNum) # represent unknown write
                                elif(syscall_name in self.target_syscall_r):
                                    syscall_idx = len(map_syscallName_to_syscallNum)+1 # represent unknown read
                                else:
                                    syscall_idx = len(map_syscallName_to_syscallNum)+2 # represent unknown syscall
                                print(key,syscall_idx,syscall_cnt,thread_num)
                            else:
                                syscall_idx = map_syscallName_to_syscallNum[syscall_name]
                        # store its normalized syscall_idx by thread ID
                        map_threadN_to_syscall[tid].append(syscall_idx/400)
            # do n-gram
            for key in map_threadN_to_syscall.keys():
                for i in range(len(map_threadN_to_syscall[key])-self.seq_len+1):
                    normal_syscall_seq.append(map_threadN_to_syscall[key][i:i+self.seq_len])
                #normal_syscall_seq.append([-1]*self.seq_len)

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
            elif(file_cnt < int(len(normal_file_info_list)*(self.train_ratio+self.validation_ratio))):
                if((file_cnt+1) % self.save_file_intvl == 0 
                    or file_cnt == int(len(normal_file_info_list)*(self.train_ratio+self.validation_ratio))-1
                ):
                    # normal_syscall_seq (thread major)
                    normal_syscall_seq = np.array(normal_syscall_seq)
                    np.save(os.path.join(dir_path,'valid_normal_'+str(valid_normal_cnt)),normal_syscall_seq)
                    print('{} shape = {}, {}'.format('valid_normal_'+str(valid_normal_cnt),normal_syscall_seq.shape,file_info[1]))
                    normal_syscall_seq = [] # clear normal_syscall_seq
                    valid_normal_cnt += 1
                else:
                    continue
            else:
                # normal_syscall_seq (thread major)
                normal_syscall_seq = np.array(normal_syscall_seq)
                np.save(os.path.join(dir_path,'test_normal_'+str(test_normal_cnt)),normal_syscall_seq)
                print('{} shape = {}, {}'.format('test_normal_'+str(test_normal_cnt),normal_syscall_seq.shape,file_info[1]))
                normal_syscall_seq = [] # clear normal_syscall_seq

                # cnt++
                test_normal_cnt += 1

        
        # get attack data
        attack_syscall_seq = []
        attack_cnt = 0
        for file_cnt, file_info in enumerate(attack_file_info_list):
            f = open(os.path.join(dir_path,file_info[1]))
            map_threadN_to_syscall = collections.OrderedDict() # put all syscalls with same thread# into a list
            #map_threadN_to_syscall[-1] = [] # for storing syscall in time order
            syscall_cnt = 0
            map_threadN_to_records = collections.OrderedDict()
            # categorize records by thread ID
            for line in f.readlines():
                features = line.split(' ')
                thread_num = features[5]
                if(thread_num not in map_threadN_to_records.keys()):
                    map_threadN_to_records[thread_num] = []
                map_threadN_to_records[thread_num].append(line)
            
            # convert syscall records to syscall number
            for tid in map_threadN_to_records.keys():
                map_threadN_to_syscall[tid] = []
                t_list = map_threadN_to_records[tid] # records list of thread_num
                for line_n in range(len(t_list)):
                    features = t_list[line_n].split(' ') 
                    syscall_direction = features[6]
                    syscall_name = features[7]
                    if(syscall_direction == '>' and syscall_name in map_syscallName_to_syscallNum.keys()):
                        # get key
                        syscall_cnt += 1
                        if(syscall_name in self.target_syscall_w or syscall_name in self.target_syscall_r):
                            next_line_features = t_list[line_n+1].split(' ')
                            key = self.get_syscall_key(features,next_line_features)
                        else:
                            key = self.get_syscall_key(features)
                        # check if key in database, and assign it a syscall index
                        if(key not in set_syscallArg):
                            if(syscall_name in self.target_syscall_w):
                                syscall_idx = len(map_syscallName_to_syscallNum) # represent unknown write
                            elif(syscall_name in self.target_syscall_r):
                                syscall_idx = len(map_syscallName_to_syscallNum)+1 # represent unknown read
                            else:
                                syscall_idx = len(map_syscallName_to_syscallNum)+2 # represent unknown syscall
                            print(key,syscall_idx,syscall_cnt,thread_num)
                        else:
                            syscall_idx = map_syscallName_to_syscallNum[syscall_name]
                        # store its normalized syscall_idx by thread ID
                        map_threadN_to_syscall[tid].append(syscall_idx/400)
            # do n-gram
            for key in map_threadN_to_syscall.keys():
                for i in range(len(map_threadN_to_syscall[key])-self.seq_len+1):
                    attack_syscall_seq.append(map_threadN_to_syscall[key][i:i+self.seq_len])
                #attack_syscall_seq.append([-1]*self.seq_len)

            # save each file's syscall
            attack_syscall_seq = np.array(attack_syscall_seq) # list to np.array
            np.save(os.path.join(dir_path,'test_attack_'+str(attack_cnt)),attack_syscall_seq) # save np.array
            print('{} shape = {}, {}'.format('test_attack_'+str(attack_cnt),attack_syscall_seq.shape,file_info[1]))
            attack_syscall_seq = []

            # cnt++
            attack_cnt += 1

        # print file count of each category
        print("Train_cnt: {}, Valid_normal_cnt: {},Test_normal_cnt: {}, Test_attack_cnt: {}".format(train_cnt,valid_normal_cnt,test_normal_cnt,attack_cnt))

        # save map_syscallArg_to_idx size
        # map_size = np.array([len(map_syscallArg_to_idx)+3]) # 3: unknown read, unknown write, unknown syscall
        # print('Map size = {}'.format(map_size[0]))
        # np.save(os.path.join(dir_path,'map_size'),map_size)
        print('Arg set size: {}'.format(len(set_syscallArg)))