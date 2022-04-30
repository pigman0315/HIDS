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
            key += ret_value + '_'
            key += file_name
        return key
        

    def process_data(self,dir_path):
        # read syscall list
        map_syscallName_to_syscallNum = {}
        f = open('./syscall_list.txt')
        next(f) # skip header
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
        ordered_syscall_seq = []
        train_cnt = 0
        valid_normal_cnt = 0
        test_normal_cnt = 0
        set_syscallArg = set() # set 'syscall_name+ret_value+file_name' as a unique key
        set_syscallArg.add('read__')
        set_syscallArg.add('write__')
        for file_cnt, file_info in enumerate(normal_file_info_list):
            #print('map size: {}'.format(len(map_syscallArg_to_idx)))
            f = open(os.path.join(dir_path,file_info[1]))
            map_threadN_to_syscall = collections.OrderedDict() # put all syscalls with same thread# into a list
            #map_threadN_to_syscall[-1] = [] # for storing syscall in time order
            syscall_cnt = 0
            for line in f.readlines():
                features = line.split(' ')
                syscall_name = features[7]
                syscall_direction = features[6]
                thread_num = features[5]
                if(syscall_direction == '>' and syscall_name in map_syscallName_to_syscallNum.keys()):
                    syscall_cnt += 1
                    # get index in syscall_map with key 'syscall+ret_val+file'
                    key = self.get_syscall_key(features)
                    syscall_idx = -100000 
                    if(file_cnt < int(len(normal_file_info_list)*self.train_ratio)): # training file
                        if(key not in set_syscallArg):
                            print(key)
                            set_syscallArg.add(key)
                        syscall_idx = map_syscallName_to_syscallNum[syscall_name]
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
                        
                    # threadID major
                    if(thread_num not in map_threadN_to_syscall.keys()):
                        map_threadN_to_syscall[thread_num] = []
                    map_threadN_to_syscall[thread_num].append(syscall_idx/400)  

                    # interleaving major
                    #map_threadN_to_syscall[-1].append(syscall_idx/400)
            f.close()
            for key in map_threadN_to_syscall.keys():
                if(key != -1):
                    if(len(map_threadN_to_syscall[key]) < self.seq_len):
                        # l = [-1]*self.seq_len
                        # l[:len(map_threadN_to_syscall[key])] = map_threadN_to_syscall[key]
                        # #print(l)
                        # normal_syscall_seq.append(l)
                        for i in range(len(map_threadN_to_syscall[key])-self.seq_len+1):
                            normal_syscall_seq.append(map_threadN_to_syscall[key][i:i+self.seq_len])
                    else:
                        for i in range(len(map_threadN_to_syscall[key])-self.seq_len+1):
                            normal_syscall_seq.append(map_threadN_to_syscall[key][i:i+self.seq_len])
                else:
                    for i in range(len(map_threadN_to_syscall[-1])-self.seq_len+1):
                        ordered_syscall_seq.append(map_threadN_to_syscall[-1][i:i+self.seq_len])
            
            # training data, save data for every "self.save_file_intvl"
            if(file_cnt < int(len(normal_file_info_list)*self.train_ratio)):
                ordered_syscall_seq = []
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

                # ordered_syscall_seq (interleaving)
                # ordered_syscall_seq = np.array(ordered_syscall_seq)
                # np.save(os.path.join(dir_path,'test_normal_ordered_'+str(test_normal_cnt)),ordered_syscall_seq)
                # print('{} shape = {}, {}'.format('test_normal_ordered_'+str(test_normal_cnt),ordered_syscall_seq.shape,file_info[1]))
                # ordered_syscall_seq = [] # clear ordered_syscall_seq

                # cnt++
                test_normal_cnt += 1

        
        # get attack data
        attack_syscall_seq = []
        ordered_syscall_seq = []
        attack_cnt = 0
        for file_cnt, file_info in enumerate(attack_file_info_list):
            f = open(os.path.join(dir_path,file_info[1]))
            map_threadN_to_syscall = collections.OrderedDict() # put all syscalls with same thread# into a list
            #map_threadN_to_syscall[-1] = [] # for storing syscall in time order
            syscall_cnt = 0
            for line in f.readlines():
                features = line.split(' ')
                syscall_name = features[7]
                syscall_direction = features[6]
                thread_num = features[5]
                if(syscall_direction == '>' and syscall_name in map_syscallName_to_syscallNum.keys()):
                    syscall_cnt += 1
                    # get index in syscall_map with key 'syscall+ret_val+file'
                    key = self.get_syscall_key(features)
                    syscall_idx = -100000 
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
                        
                    # threadID major
                    if(thread_num not in map_threadN_to_syscall.keys()):
                        map_threadN_to_syscall[thread_num] = []
                    map_threadN_to_syscall[thread_num].append(syscall_idx/400)  

                    # interleaving major
                    #map_threadN_to_syscall[-1].append(syscall_idx/400)
            f.close()
            for key in map_threadN_to_syscall.keys():
                if(key != -1):
                    if(len(map_threadN_to_syscall[key]) < self.seq_len):
                        l = [-1]*self.seq_len
                        l[:len(map_threadN_to_syscall[key])] = map_threadN_to_syscall[key]
                        #print(l)
                        attack_syscall_seq.append(l)
                    else:
                        for i in range(len(map_threadN_to_syscall[key])-self.seq_len+1):
                            attack_syscall_seq.append(map_threadN_to_syscall[key][i:i+self.seq_len])
                else:
                    for i in range(len(map_threadN_to_syscall[-1])-self.seq_len+1):
                        ordered_syscall_seq.append(map_threadN_to_syscall[-1][i:i+self.seq_len])
            # save each file's syscall
            attack_syscall_seq = np.array(attack_syscall_seq) # list to np.array
            np.save(os.path.join(dir_path,'test_attack_'+str(attack_cnt)),attack_syscall_seq) # save np.array
            print('{} shape = {}, {}'.format('test_attack_'+str(attack_cnt),attack_syscall_seq.shape,file_info[1]))
            attack_syscall_seq = []

            # ordered_syscall_seq (interleaving)
            # ordered_syscall_seq = np.array(ordered_syscall_seq)
            # np.save(os.path.join(dir_path,'test_attack_ordered_'+str(attack_cnt)),ordered_syscall_seq)
            # print('{} shape = {}, {}'.format('test_attack_ordered_'+str(attack_cnt),ordered_syscall_seq.shape,file_info[1]))
            # ordered_syscall_seq = [] # clear ordered_syscall_seq

            # cnt++
            attack_cnt += 1

        # print file count of each category
        print("Train_cnt: {}, Valid_normal_cnt: {},Test_normal_cnt: {}, Test_attack_cnt: {}".format(train_cnt,valid_normal_cnt,test_normal_cnt,attack_cnt))

        # save map_syscallArg_to_idx size
        # map_size = np.array([len(map_syscallArg_to_idx)+3]) # 3: unknown read, unknown write, unknown syscall
        # print('Map size = {}'.format(map_size[0]))
        # np.save(os.path.join(dir_path,'map_size'),map_size)
        print('Arg set size: {}'.format(len(set_syscallArg)))