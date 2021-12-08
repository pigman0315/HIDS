import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sklearn
from sklearn.model_selection import train_test_split
from LID_preprocess import Preprocess
from LID_model import AE
import re

# Globla variables
NEED_PREPROCESS = True
ROOT_DIR = '../../LID-DS/'
TARGET_DIR = 'CVE-2018-3760'
INPUT_DIR = ROOT_DIR+TARGET_DIR
SEQ_LEN = 5
TRAIN_RATIO = 0.2 # ratio between size of training data and validation data
EPOCHS = 10 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 128 # batch size for training
HIDDEN_SIZE = 256 # encoder's 1st lstm layer hidden size 
DROP_OUT = 0.0
VEC_LEN = 16 # length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
LOG_INTERVAL = 1000 # log interval of printing message
SAVE_FILE_INTVL = 50 # save file interval

def get_npy_list(type):
    file_list = os.listdir(INPUT_DIR)
    find_pattern = re.compile(rf"{type}_[0-9]*\.npy") # $type_
    npy_list = find_pattern.findall(' '.join(file_list))
    print(npy_list)
    return npy_list

def train(model):
    # get train_*.npy file list
    npy_list = get_npy_list(type='train')

    # training
    #model.load_state_dict(torch.load('weight.pth')) # get pre-trained model
    train_loss_list = []
    for epoch in range(EPOCHS):
        loss = 0
        for npy_file in npy_list:
            train_data = np.load(os.path.join(INPUT_DIR,npy_file))
            train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
            for i, x in enumerate(train_dataloader):
                # feed forward
                x = x.float()
                x = x.view(-1, SEQ_LEN, VEC_LEN)
                x = x.to(device)
                result = model(x)
                
                # backpropagation
                loss = criterion(result, x)
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
    torch.save(model.state_dict(), "./weight_"+TARGET_DIR+'_'+str(EPOCHS)+".pth")
def validation(model):
    # get validation_*.npy file list
    npy_list = get_npy_list(type='validation')

    # validation
    validation_loss = 0
    validation_loss_list = []
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    with torch.no_grad():
        for npy_file in npy_list:   
            validation_data = np.load(os.path.join(INPUT_DIR,npy_file))
            validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
            for i, x in enumerate(validation_dataloader):
                # feed forward
                x = x.float()
                x = x.view(-1, SEQ_LEN, VEC_LEN)
                x = x.to(device)
                result = model(x)
                
                # calculate loss
                validation_loss = criterion(result, x)
                validation_loss_list.append(validation_loss.item())
                
                # print progress
                if(i % LOG_INTERVAL == 0):
                    print('{}/{}, loss = {}'.format(i,len(validation_data)//BATCH_SIZE,validation_loss))
    print('=== Validation Avg. Loss:',sum(validation_loss_list)/len(validation_loss_list),'===')

def test_attack_data(model):
    # get attack_*.npy file list
    npy_list = get_npy_list(type='attack')

    # test attack data
    attack_loss = 0
    attack_loss_list = []
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    with torch.no_grad():
        for npy_file in npy_list:
            attack_data = np.load(os.path.join(INPUT_DIR,npy_file))
            attack_dataloader = DataLoader(attack_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
            for i, x in enumerate(attack_dataloader):
                # feed forward
                x = x.float()
                x = x.view(-1, SEQ_LEN, VEC_LEN)
                x = x.to(device)
                result = model(x)
                
                # calculate loss
                attack_loss = criterion(result, x)
                attack_loss_list.append(attack_loss.item())

                # print progress
                if(i % LOG_INTERVAL == 0):
                    print('{}/{}, loss = {}'.format(i,len(attack_data)//BATCH_SIZE,attack_loss))
            
        print('=== Attack avg loss = {} ==='.format(sum(attack_loss_list)/len(attack_loss_list)))


if __name__ == '__main__':  
    # Check if using GPU
    print("Is using GPU?",torch.cuda.is_available())
    device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:",device)
    print("Currently using GPU:",torch.cuda.get_device_name(0))
    
    # preprocess row data into .npy file
    if(NEED_PREPROCESS):
        prep = Preprocess(seq_len=SEQ_LEN,train_ratio=TRAIN_RATIO,save_file_intvl=SAVE_FILE_INTVL)
        prep.process_data(INPUT_DIR)

    # model setting
    model = AE(seq_len=SEQ_LEN,vec_len=VEC_LEN,hidden_size=HIDDEN_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # train
    train(model)

    validation(model)

    test_attack_data(model)



        
