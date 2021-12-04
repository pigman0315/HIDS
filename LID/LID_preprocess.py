import os
import sys
from datetime import datetime
import numpy as np

class Preprocess:
    def __init__(self,seq_len=20,train_ratio=0.2):
        self.seq_len = seq_len
        self.train_ratio = train_ratio

    def filtering_and_abstraction(self,traces): # NOT Done yet
        filter_list = ['17','20','21','22','31','32','33','34','61','62','72','73','81','85','86',
                   '87','88','95','98' ,'100','101','102','103','107','108','109','110','111',
                   '112','113','114','115','120','121','123','125','126','127','141','143','144',
                   '145','146','147','148','149','150','151','152','153','154','155','156','157',
                   '159' ,'158','163','165','168','169','170','171','172','173','174' ,'175' ,'176',
                   '177','178','179','183','185','204','205' ,'213','214','215', '216' ,'222' ,'226',
                   '227' ,'228','229','230','237','260','266','278']
        abstract = []
        for system_calls in range(len(traces)):
            if traces[system_calls] in filter_list:
                continue
            if traces[system_calls] == ('6'):
                abstract.append('5')
            elif traces[system_calls] == ('7'):
                abstract.append('5')
            elif traces[system_calls] == ('9'):
                abstract.append('8')
            elif traces[system_calls] == ('10'):
                abstract.append('8')
            elif traces[system_calls] == ('12'):
                abstract.append('11')
            elif traces[system_calls] == ('13'):
                abstract.append('11')
            elif traces[system_calls] == ('15'):
                abstract.append('14')
            elif traces[system_calls] == ('16'):
                abstract.append('14')
            elif traces[system_calls] == ('24'):
                abstract.append('23')
            elif traces[system_calls] == ('46'):
                abstract.append('45')
            elif traces[system_calls] == ('50'):
                abstract.append('49')
            elif traces[system_calls] == ('53'):
                abstract.append('52')
            elif traces[system_calls] == ('55'):
                abstract.append('54')
            elif traces[system_calls] == ('65'):
                abstract.append('63')
            elif traces[system_calls] == ('67'):
                abstract.append('63')
            elif traces[system_calls] == ('69'):
                abstract.append('63')
            elif traces[system_calls] == ('66'):
                abstract.append('64')
            elif traces[system_calls] == ('68'):
                abstract.append('64')
            elif traces[system_calls] == ('70'):
                abstract.append('64')
            elif traces[system_calls] == ('80'):
                abstract.append('79')
            elif traces[system_calls] == ('76'):
                abstract.append('75')
            elif traces[system_calls] == ('83'):
                abstract.append('82')
            elif traces[system_calls] == ('84'):
                abstract.append('82')
            elif traces[system_calls] == ('94'):
                abstract.append('93')
            elif traces[system_calls] == ('131'):
                abstract.append('130')
            elif traces[system_calls] == ('205'):
                abstract.append('204')
            elif traces[system_calls] == ('242'):
                abstract.append('202')
            elif traces[system_calls] == ('269'):
                abstract.append('211')
            elif traces[system_calls] == ('276'):
                abstract.append('38')
            elif traces[system_calls] == ('284'):
                abstract.append('228')
            else:
                abstract.append(traces[system_calls])
        return abstract
    
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
                    all_syscall_list.append(syscall_num_map[features[7]])
            f.close()
            all_syscall_list = self.filtering_and_abstraction(all_syscall_list)
            for i in range(len(all_syscall_list)-self.seq_len+1):
                 normal_syscall_seq.append([float(syscall)/len(syscall_num_map) for syscall in all_syscall_list[i:i+self.seq_len]])
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
                if(features[6] == '>' and time_delta > exploit_time and features[7] in syscall_num_map.keys()):
                    all_syscall_list.append(syscall_num_map[features[7]])
            f.close()
            all_syscall_list = self.filtering_and_abstraction(all_syscall_list)
            for i in range(len(all_syscall_list)-self.seq_len+1):
                attack_syscall_seq.append([float(syscall)/len(syscall_num_map) for syscall in all_syscall_list[i:i+self.seq_len]])
            
        attack_syscall_seq = np.array(attack_syscall_seq) # list to np.array
        np.save(os.path.join(dir_path,'attack'),attack_syscall_seq) # save np.array
        print(attack_syscall_seq.shape)