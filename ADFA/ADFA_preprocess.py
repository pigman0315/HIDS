import os
import numpy as np
import torch

class Preprocess:
    def __init__(self,seq_len=20,total_syscall_num=264):
        # sequence length of syscalls(sliding window size)
        self.seq_len = seq_len
        self.total_syscall_num = total_syscall_num

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

    def get_syscall_sequence_list(self,dir_path):
        syscall_seq_list = []
        files = os.listdir(dir_path)
        for file_name in files:
            f = open(os.path.join(dir_path,file_name))
            syscall_list = f.readline().split(' ')
            syscall_list = self.filtering_and_abstraction(syscall_list)
            for i in range(len(syscall_list)-self.seq_len):
                syscall_seq_list.append([float(syscall)/self.total_syscall_num for syscall in syscall_list[i:i+self.seq_len]])
        return syscall_seq_list

    def read_files(self,input_dir):
        TRAIN_DIR_PATH = os.path.join(input_dir,"Training_Data_Master")
        VALIDATION_DIR_PATH = os.path.join(input_dir,"Validation_Data_Master")
        ATTACK_DIR_PATH = os.path.join(input_dir,"Attack_Data_Master")

        # read training data
        self.train_data = []
        syscall_seq_list = self.get_syscall_sequence_list(TRAIN_DIR_PATH)
        self.train_data = syscall_seq_list
        self.train_data = np.array(self.train_data)
        print(self.train_data.shape)

        # read validation data
        self.validation_data = []
        validation_files = os.listdir(VALIDATION_DIR_PATH)
        syscall_seq_list = self.get_syscall_sequence_list(VALIDATION_DIR_PATH)
        self.validation_data = syscall_seq_list
        self.validation_data = np.array(self.validation_data)
        print(self.validation_data.shape)

        # read attack data
        self.attack_dict = {}
        self.attack_dict['Adduser'] = []
        self.attack_dict['Hydra_FTP'] = []
        self.attack_dict['Hydra_SSH'] = []
        self.attack_dict['Java_Meterpreter'] = []
        self.attack_dict['Meterpreter'] = []
        self.attack_dict['Web_Shell'] = []
        attack_dirs = os.listdir(ATTACK_DIR_PATH)
        for attack_dir in attack_dirs:
            if('Adduser' in attack_dir):
                syscall_seq_list = self.get_syscall_sequence_list(os.path.join(ATTACK_DIR_PATH,attack_dir))
                for syscall_seq in syscall_seq_list:
                    self.attack_dict['Adduser'].append(syscall_seq)
            elif('Hydra_FTP' in attack_dir):
                syscall_seq_list = self.get_syscall_sequence_list(os.path.join(ATTACK_DIR_PATH,attack_dir))
                for syscall_seq in syscall_seq_list:
                    self.attack_dict['Hydra_FTP'].append(syscall_seq)
            elif('Hydra_SSH' in attack_dir):
                syscall_seq_list = self.get_syscall_sequence_list(os.path.join(ATTACK_DIR_PATH,attack_dir))
                for syscall_seq in syscall_seq_list:
                    self.attack_dict['Hydra_SSH'].append(syscall_seq)
            elif('Java_Meterpreter' in attack_dir):
                syscall_seq_list = self.get_syscall_sequence_list(os.path.join(ATTACK_DIR_PATH,attack_dir))
                for syscall_seq in syscall_seq_list:
                    self.attack_dict['Java_Meterpreter'].append(syscall_seq)
            elif('Meterpreter' in attack_dir):
                syscall_seq_list = self.get_syscall_sequence_list(os.path.join(ATTACK_DIR_PATH,attack_dir))
                for syscall_seq in syscall_seq_list:
                    self.attack_dict['Meterpreter'].append(syscall_seq)
            elif('Web_Shell' in attack_dir):
                syscall_seq_list = self.get_syscall_sequence_list(os.path.join(ATTACK_DIR_PATH,attack_dir))
                for syscall_seq in syscall_seq_list:
                    self.attack_dict['Web_Shell'].append(syscall_seq)
        self.attack_dict['Adduser'] = np.array(self.attack_dict['Adduser'])
        self.attack_dict['Hydra_FTP'] = np.array(self.attack_dict['Hydra_FTP'])
        self.attack_dict['Hydra_SSH'] = np.array(self.attack_dict['Hydra_SSH'])
        self.attack_dict['Java_Meterpreter'] = np.array(self.attack_dict['Java_Meterpreter'])
        self.attack_dict['Meterpreter'] = np.array(self.attack_dict['Meterpreter'])
        self.attack_dict['Web_Shell'] = np.array(self.attack_dict['Web_Shell'])
        print(self.attack_dict['Adduser'].shape)
        print(self.attack_dict['Hydra_FTP'].shape)
        print(self.attack_dict['Hydra_SSH'].shape)
        print(self.attack_dict['Java_Meterpreter'].shape)
        print(self.attack_dict['Meterpreter'].shape)
        print(self.attack_dict['Web_Shell'].shape)

    def output_files(self,output_dir):
        # output syscall sequence list of training data
        np.save(os.path.join(output_dir,'train.npy'),self.train_data)

        # output syscall sequence list of validation data
        np.save(os.path.join(output_dir,'validation.npy'),self.validation_data)

        # output syscall sequence list of attack data
        for key in self.attack_dict.keys():
            np.save(os.path.join(output_dir,key+'.npy'),self.attack_dict[key])