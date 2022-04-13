import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sklearn
from sklearn.model_selection import train_test_split
from LID_preprocess import Preprocess
from LID_model import CMAE
from memory_module import EntropyLossEncap
class SuspiciousCounter:
    def __init__(self,threshold):
        self.queue = []
        self.count = 0
        self.threshold = threshold
    def push(self,anomaly_score):
        self.queue.append(anomaly_score)
        if(anomaly_score > self.threshold):
            self.count += 1
    def pop(self):
        if(self.queue[0] > self.threshold):
            self.count -= 1
        del self.queue[0]

def get_npy_list(type):
    file_list = os.listdir(INPUT_DIR)
    find_pattern = re.compile(rf"{type}_[0-9]*\.npy") # $type_
    npy_list = find_pattern.findall(' '.join(file_list))
    #print(npy_list)
    return npy_list

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    entropy_loss_func = EntropyLossEncap().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # get train_*.npy file list
    npy_list = get_npy_list(type='train')
    
    if(NEED_TRAIN):
        # training
        #model.load_state_dict(torch.load('weight.pth')) # get pre-trained model
        train_loss_list = []
        for epoch in range(EPOCHS):
            loss = 0
            for npy_file in npy_list:
                train_data = np.load(os.path.join(INPUT_DIR,npy_file))
                train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)
                for i, x in enumerate(train_dataloader):
                    # feed forward
                    x = x.float()
                    x = x.view(-1, SEQ_LEN, VEC_LEN)
                    x = x.to(device)
                    result, atten_weight = model(x)
            
                    # backpropagation
                    x = x.view(-1,SEQ_LEN,VEC_LEN)
                    reconstr_loss = criterion(result, x)
                    entropy_loss = entropy_loss_func(atten_weight)
                    loss = reconstr_loss + ENTROPY_LOSS_WEIGHT * entropy_loss
                    #loss = reconstr_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # print progress
                    if(i % LOG_INTERVAL == 0):
                        print('Epoch {}({}/{}), loss = {}'.format(epoch+1,i,len(train_data)//BATCH_SIZE,loss))
                
                # record last epoch's loss
                if(epoch == EPOCHS-1):
                    train_loss_list.append(loss.item())
            print('=== epoch: {}, loss: {} ==='.format(epoch+1,loss))
            torch.save(model.state_dict(), "./weight.pth")
        print('=== Train Avg. Loss:',sum(train_loss_list)/len(train_loss_list),'===')
        #torch.save(model.state_dict(), "./weight_"+TARGET_DIR+'_'+str(EPOCHS)+"_MemAE"+".pth")

    # get threshold to distinguish normal and attack data
    if(MAX_LOSS == None):
        model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
        criterion_none = nn.MSELoss(reduction='none')
        model.eval()
        max_loss = 0
        with torch.no_grad():
            loss_list = []
            for npy_file in npy_list:
                print('Scanning {}'.format(npy_file))
                train_data = np.load(os.path.join(INPUT_DIR,npy_file))
                train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)
                for i, x in enumerate(train_dataloader):
                    # feed forward
                    x = x.float()
                    x = x.view(-1, SEQ_LEN, VEC_LEN)
                    x = x.to(device)
                    result,_ = model(x)
                    # calculate loss
                    loss_mat = criterion_none(result, x)
                    loss_mat = loss_mat.to('cpu')
                    for loss in loss_mat:
                        loss_1D = torch.flatten(loss).tolist()
                        loss_sum = sum(loss_1D)
                        loss_list.append(loss_sum)
            loss_list.sort()
            max_loss = loss_list[int(len(loss_list)*THRESHOLD_PERCENTILE)]
        #threshold = max_loss*(THRESHOLD_RATIO)
        return max_loss
    else:
        return MAX_LOSS

def test(model,threshold):
    # matrics
    fp = 0
    tp = 0
    fn = 0
    tn = 0

    # model setting
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
    model.eval()
    criterion_none = nn.MSELoss(reduction='none') # loss function

    with torch.no_grad():
        npy_list = get_npy_list(type='test_normal') # get validation_*.npy file list
        # for each file
        for file_num, npy_file in enumerate(npy_list):
            print('Testing {}'.format(npy_file))
            validation_data = np.load(os.path.join(INPUT_DIR,npy_file))
            validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE,shuffle=False)
            suspicious_counter = 0 # for new detect algo.
            #suspicious_counter = SuspiciousCounter(threshold) # for old detect algo.
            is_attack = False
            for i, x in enumerate(validation_dataloader):
                # feed forward
                x = x.float()
                x = x.view(-1, SEQ_LEN, VEC_LEN)
                x = x.to(device)
                result,atten_weight = model(x)
                
                # calculate loss
                loss_mat = criterion_none(result, x)
                loss_mat = loss_mat.to('cpu')
                for loss in loss_mat:
                    loss_1D = torch.flatten(loss).tolist()
                    loss_sum = sum(loss_1D)

                    ### Old detection algo.
                    # suspicious_counter.push(loss_sum)
                    # if(len(suspicious_counter.queue) >= QUEUE_LEN):
                    #     suspicious_counter.pop()
                    # if(suspicious_counter.count > SUSPICIOUS_THRESHOLD):
                    #     is_attack = True
                    #     break

                    ### New detection algo.
                    if(loss_sum > threshold):
                        suspicious_counter += 1
                    else:
                        suspicious_counter -= 1
                        suspicious_counter = max(0,suspicious_counter)
                    if(suspicious_counter >= SUSPICIOUS_THRESHOLD):
                        is_attack = True
                        break
                if(is_attack == True):
                    break
            if(is_attack == True):
                fp += 1
            else:
                tn += 1
                
    # test attack data
    npy_list = get_npy_list(type='test_attack') # get attack_*.npy file list
    with torch.no_grad():
        for file_num, npy_file in enumerate(npy_list):
            print('Testing {}'.format(npy_file))
            attack_data = np.load(os.path.join(INPUT_DIR,npy_file))
            attack_dataloader = DataLoader(attack_data, batch_size=BATCH_SIZE,shuffle=False)
            suspicious_counter = 0 # for new detect algo.
            #suspicious_counter = SuspiciousCounter(threshold) # for old detect algo.
            is_attack = False
            for i, x in enumerate(attack_dataloader):
                # feed forward
                x = x.float()
                x = x.view(-1, SEQ_LEN, VEC_LEN)
                x = x.to(device)
                result,atten_weight = model(x)
                
                # calculate loss
                loss_mat = criterion_none(result, x)
                loss_mat = loss_mat.to('cpu')
                for loss in loss_mat:
                    loss_1D = torch.flatten(loss).tolist()
                    loss_sum = sum(loss_1D)

                    ### Old detection algo.
                    # suspicious_counter.push(loss_sum)
                    # if(len(suspicious_counter.queue) >= QUEUE_LEN):
                    #     suspicious_counter.pop()
                    # if(suspicious_counter.count > SUSPICIOUS_THRESHOLD):
                    #     is_attack = True
                    #     break

                    ### New detection algo.
                    if(loss_sum > threshold):
                        suspicious_counter += 1
                    else:
                        suspicious_counter -= 1
                        suspicious_counter = max(0,suspicious_counter)
                    if(suspicious_counter >= SUSPICIOUS_THRESHOLD):
                        is_attack = True
                        break
                if(is_attack == True):
                    break
            if(is_attack == True):
                tp += 1
            else:
                fn += 1
    
    # show results
    print('=== Results ===')
    print('tp = {}, tn = {}, fp = {}, fn = {}'.format(tp,tn,fp,fn))
    print('Accuracy = {}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Precision = {}'.format(tp/(tp+fp)))
    print('Recall = {}'.format(tp/(tp+fn)))
    print('F1-score = {}'.format(2*tp/(2*tp+fp+fn)))
    print('==============')

def check_counter(model,threshold):
    # model setting
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
    model.eval()
    criterion_none = nn.MSELoss(reduction='none') # loss function

    # with torch.no_grad():
    #     npy_list = get_npy_list(type='test_normal') # get validation_*.npy file list
    #     # for each file
    #     for file_num, npy_file in enumerate(npy_list):
    #         print('Testing {}'.format(npy_file))
    #         validation_data = np.load(os.path.join(INPUT_DIR,npy_file))
    #         validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE,shuffle=False)
    #         suspicious_counter = 0 # for new detect algo.
    #         sc_list = []
    #         for i, x in enumerate(validation_dataloader):
    #             # feed forward
    #             x = x.float()
    #             x = x.view(-1, SEQ_LEN, VEC_LEN)
    #             x = x.to(device)
    #             result,atten_weight = model(x)
                
    #             # calculate loss
    #             loss_mat = criterion_none(result, x)
    #             loss_mat = loss_mat.to('cpu')
    #             for loss in loss_mat:
    #                 loss_1D = torch.flatten(loss).tolist()
    #                 loss_sum = sum(loss_1D)
    #                 ### New detection algo.
    #                 if(loss_sum > threshold):
    #                     suspicious_counter += 1
    #                 else:
    #                     suspicious_counter -= 1
    #                     suspicious_counter = max(0,suspicious_counter)
    #                 sc_list.append(suspicious_counter)
    #         plt.plot(sc_list)
    #         plt.savefig(os.path.join(INPUT_DIR,npy_file[:-4]+'.png')) 
            # plt.close()
            # if(file_num == 5):
            #     exit()
                
    # test attack data
    npy_list = get_npy_list(type='test_attack') # get attack_*.npy file list
    with torch.no_grad():
        for file_num, npy_file in enumerate(npy_list):
            print('Testing {}'.format(npy_file))
            attack_data = np.load(os.path.join(INPUT_DIR,npy_file))
            attack_dataloader = DataLoader(attack_data, batch_size=BATCH_SIZE,shuffle=False)
            suspicious_counter = 0             # for new detect algo.
            sc_list = []
            for i, x in enumerate(attack_dataloader):
                # feed forward
                x = x.float()
                x = x.view(-1, SEQ_LEN, VEC_LEN)
                x = x.to(device)
                result,atten_weight = model(x)
                
                # calculate loss
                loss_mat = criterion_none(result, x)
                loss_mat = loss_mat.to('cpu')
                for loss in loss_mat:
                    loss_1D = torch.flatten(loss).tolist()
                    loss_sum = sum(loss_1D)

                    ### New detection algo.
                    if(loss_sum > threshold):
                        suspicious_counter += 1
                    else:
                        suspicious_counter -= 1
                        suspicious_counter = max(0,suspicious_counter)
                    sc_list.append(suspicious_counter)
            plt.plot(sc_list)
            plt.savefig(os.path.join(INPUT_DIR,npy_file[:-4]+'.png')) 
            plt.close()
            if(file_num == 5):
                exit()
                    

# Global variables
NEED_PREPROCESS = False
NEED_TRAIN = False
ROOT_DIR = '../../LID-DS/'
TARGET_DIR = 'CVE-2017-7529'
MODEL_WEIGHT_PATH = 'weight.pth'
INPUT_DIR = ROOT_DIR+TARGET_DIR
SEQ_LEN = 10
TRAIN_RATIO = 0.2 # ratio of training data in normal data
EPOCHS = 10 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 128 # batch size for training
HIDDEN_SIZE = 256 # encoder's 1st layer hidden size 
DROP_OUT = 0.0
VEC_LEN = 1 # length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
LOG_INTERVAL = 1000 # log interval of printing message
SAVE_FILE_INTVL = 50 # saving-file interval for training (prevent memory explosion)
THRESHOLD_RATIO = 5 # if the loss of input is higher than theshold*(THRESHOLD_RATIO), then it is considered to be suspicious
SUSPICIOUS_THRESHOLD = SEQ_LEN # if suspicious count higher than this threshold then it is considered to be an attack file
THRESHOLD_PERCENTILE = 0.8 # percentile of reconstruction error in training data
ENTROPY_LOSS_WEIGHT = 0.0002 # default: 0.0002
MEM_DIM = 200
SHRINK_THRESHOLD = 0.1/MEM_DIM # 1/MEM_DIM ~ 3/MEM_DIM
MAX_LOSS = None # to speedup experiment
#QUEUE_LEN = 10 # M in old detection algo.

if __name__ == '__main__':  
    # Check if using GPU
    print("Is using GPU?",torch.cuda.is_available())
    device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:",device)
    print("Currently using GPU:",torch.cuda.get_device_name(0))
    
    # preprocess row data into .npy file
    if(NEED_PREPROCESS):
        prep = Preprocess(seq_len=SEQ_LEN
        ,train_ratio=TRAIN_RATIO
        ,save_file_intvl=SAVE_FILE_INTVL
        )
        prep.remove_npy(INPUT_DIR)
        prep.process_data(INPUT_DIR)

    # model setting
    model = CMAE(seq_len=SEQ_LEN,vec_len=VEC_LEN,hidden_size=HIDDEN_SIZE,mem_dim=MEM_DIM,shrink_thres=SHRINK_THRESHOLD).to(device)
    
    # train
    max_loss = train(model)
    print('Max loss = {}'.format(max_loss))
    threshold = max_loss*THRESHOLD_RATIO

    # test
    test(model,threshold)

    #check_counter(model,threshold)



        
