import os
import numpy as np
import torch

class Preprocess:
    def __init__(self,seq_len=20,total_syscall_num=264):
        # sequence length of syscalls(sliding window size)
        self.seq_len = seq_len
        self.total_syscall_num = total_syscall_num

    def filtering_and_abstraction(self,syscall_list): # NOT Done yet
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