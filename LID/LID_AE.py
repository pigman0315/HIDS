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

# Globla variables
NEED_PREPROCESS = False
INPUT_DIR = '../../LID-DS/CVE-2012-2122'
SEQ_LEN = 20
TRAIN_RATIO = 0.2 # ratio between size of training data and validation data
EPOCHS = 10 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 128 # batch size for training
HIDDEN_SIZE = 256 # encoder's 1st lstm layer hidden size 
DROP_OUT = 0.0
VEC_LEN = 1 # length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
LOG_INTERVAL = 1000 # log interval of printing message

def train(model):
    # training
    model.load_state_dict(torch.load('weight.pth')) # get pre-trained model
    train_data = np.load(os.path.join(INPUT_DIR,'train.npy'))
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    train_loss_list = []
    for epoch in range(EPOCHS):
        loss = 0
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

def validation(model):
    # validation
    validation_data = np.load(os.path.join(INPUT_DIR,'valid.npy'))
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    validation_loss = 0
    validation_loss_list = []
    with torch.no_grad():
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
# test attack data
def test_attack_data(model):
    attack_data = np.load(os.path.join(INPUT_DIR,'attack.npy'))
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    attack_dataloader = DataLoader(attack_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    attack_loss = 0
    attack_loss_list = []
    with torch.no_grad():
        for i, x in enumerate(attack_dataloader):
            # feed forward
            x = x.float()
            x = x.view(-1, SEQ_LEN, VEC_LEN)
            x = x.to(device)
            result = model(x)
            
            # calculate loss
            attack_loss = criterion(result, x)
            attack_loss_list.append(attack_loss.item())
            
        print('=== Avg loss = {} ==='.format(sum(attack_loss_list)/len(attack_loss_list)))


if __name__ == '__main__':  
    # Check if using GPU
    print("Is using GPU?",torch.cuda.is_available())
    device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:",device)
    print("Currently using GPU:",torch.cuda.get_device_name(0))
    
    # preprocess row data into .npy file
    if(NEED_PREPROCESS):
        prep = Preprocess(seq_len=SEQ_LEN,train_ratio=TRAIN_RATIO)
        prep.process_data(INPUT_DIR)

    # model setting
    model = AE(seq_len=SEQ_LEN,hidden_size=HIDDEN_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # train
    train(model)

    # validation
    validation(model)

    # test attack data
    test_attack_data(model)



        