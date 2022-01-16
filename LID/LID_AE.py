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
from LID_model import AE,CAE
import re

# Globla variables
NEED_PREPROCESS = True
NEED_TRAIN = True
ROOT_DIR = '../../LID-DS/'
TARGET_DIR = 'CVE-2017-7529'
INPUT_DIR = ROOT_DIR+TARGET_DIR
SEQ_LEN = 20
TRAIN_RATIO = 0.3 # ratio between size of training data and validation data
EPOCHS = 10 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 128 # batch size for training
HIDDEN_SIZE = 256 # encoder's 1st layer hidden size 
DROP_OUT = 0.0
VEC_LEN = 16 # length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
LOG_INTERVAL = 1000 # log interval of printing message
SAVE_FILE_INTVL = 50 # saving-file interval for training (prevent memory explosion)

def get_npy_list(type):
    file_list = os.listdir(INPUT_DIR)
    find_pattern = re.compile(rf"{type}_[0-9]*\.npy") # $type_
    npy_list = find_pattern.findall(' '.join(file_list))
    #print(npy_list)
    return npy_list

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

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
    #torch.save(model.state_dict(), "./weight_"+TARGET_DIR+'_'+str(EPOCHS)+".pth")

def validation(model):
    criterion = nn.MSELoss(reduction='none')

    # get validation_*.npy file list
    npy_list = get_npy_list(type='validation')

    # validation
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    with torch.no_grad():
        # for each file
        for file_num, npy_file in enumerate(npy_list):
            if((file_num+1)%100 == 0):
                validation_data = np.load(os.path.join(INPUT_DIR,npy_file))
                validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
                validation_loss = 0
                validation_loss_list = []
                for i, x in enumerate(validation_dataloader):
                    # feed forward
                    x = x.float()
                    x = x.view(-1, SEQ_LEN, VEC_LEN)
                    x = x.to(device)
                    result = model(x)
                    
                    # calculate loss
                    validation_loss = criterion(result, x)
                    validation_loss = validation_loss.view(-1,SEQ_LEN).to('cpu')
                    for vl in validation_loss:
                        vl = vl.tolist()
                        validation_loss_list.append(sum(vl))

                # show top loss
                validation_loss_list.sort()
                print('=== Top loss in validation data ===')
                print(validation_loss_list[-20:])
                print('===================================')

def test_attack_data(model):
    criterion = nn.MSELoss(reduction='none')

    # get attack_*.npy file list
    npy_list = get_npy_list(type='attack')

    # test attack data
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    with torch.no_grad():
        for file_num, npy_file in enumerate(npy_list):
            if((file_num+1)%20 == 0):
                attack_data = np.load(os.path.join(INPUT_DIR,npy_file))
                attack_dataloader = DataLoader(attack_data, batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
                attack_loss = 0
                attack_loss_list = []
                for i, x in enumerate(attack_dataloader):
                    # feed forward
                    x = x.float()
                    x = x.view(-1, SEQ_LEN, VEC_LEN)
                    x = x.to(device)
                    result = model(x)
                    
                    # calculate loss
                    attack_loss = criterion(result, x)
                    attack_loss = attack_loss.view(-1,SEQ_LEN).to('cpu')
                    for al in attack_loss:
                        al = al.tolist()
                        attack_loss_list.append(sum(al))
                
                # show top loss
                attack_loss_list.sort()
                print('=== Top loss in attack data ===')
                print(attack_loss_list[-20:])
                print('===================================')


if __name__ == '__main__':  
    # Check if using GPU
    print("Is using GPU?",torch.cuda.is_available())
    device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:",device)
    print("Currently using GPU:",torch.cuda.get_device_name(0))
    
    # preprocess row data into .npy file
    if(NEED_PREPROCESS):
        prep = Preprocess(seq_len=SEQ_LEN,train_ratio=TRAIN_RATIO,save_file_intvl=SAVE_FILE_INTVL)
        prep.remove_npy(INPUT_DIR)
        prep.process_data(INPUT_DIR)

    # model setting
    model = CAE(seq_len=SEQ_LEN,vec_len=VEC_LEN,hidden_size=HIDDEN_SIZE).to(device)

    if(NEED_TRAIN == True):
        # train
        train(model)
    
    # validation
    validation(model)

    # test
    test_attack_data(model)



        
