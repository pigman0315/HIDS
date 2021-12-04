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
from LID_model import CMAE
from memory_module import EntropyLossEncap

# Globla variables
NEED_PREPROCESS = True
ROOT_DIR = '../../LID-DS/'
TARGET_DIR = 'CVE-2014-0160'
INPUT_DIR = ROOT_DIR+TARGET_DIR
TRAIN_RATIO = 0.2 # ratio between size of training data and validation data
SEQ_LEN = 1024 # n-gram length
SEQ_LEN_sqrt = 12
TOTAL_SYSCALL_NUM = 334
EPOCHS = 10 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 128 # batch size for training
HIDDEN_SIZE = 256 # encoder's 1st layer size 
DROPOUT = 0.0
VEC_LEN = 1 # length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
LOG_INTERVAL = 1000 # log interval of printing message
ENTROPY_LOSS_WEIGHT = 0.0002
MEM_DIM = 200
SHRINK_THRESHOLD = 1/MEM_DIM # 1/MEM_DIM ~ 3/MEM_DIM

def train(model):
    # training
    #model.load_state_dict(torch.load('weight.pth')) # get pre-trained model
    train_data = np.load(os.path.join(INPUT_DIR,'train.npy'))
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    train_loss_list = []
    for epoch in range(EPOCHS):
        loss = 0
        for i, x in enumerate(train_dataloader):
            # feed forward
            x = x.float()
            x = x.view(-1, SEQ_LEN, VEC_LEN)
            #x = x.view(-1, SEQ_LEN_sqrt,SEQ_LEN_sqrt)
            x = x.to(device)
            result, atten_weight = model(x)
            
            # backpropagation
            x = x.view(-1,SEQ_LEN,VEC_LEN)
            reconstr_loss = criterion(result, x)
            entropy_loss = entropy_loss_func(atten_weight)
            loss = reconstr_loss + ENTROPY_LOSS_WEIGHT * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print progress
            if(i % LOG_INTERVAL == 0):
                print('Epoch {}({}/{}), reconstr. loss = {}, entropy loss = {}'.format(epoch+1,i,len(train_data)//BATCH_SIZE,reconstr_loss,entropy_loss))
            
            # record last epoch's loss
            if(epoch == EPOCHS-1):
                train_loss_list.append(reconstr_loss.item())
        print('=== epoch: {}, reconstr. loss = {}, entropy loss = {} ==='.format(epoch+1,reconstr_loss,entropy_loss))
        torch.save(model.state_dict(), "./weight.pth")
    print('=== Train Avg. Loss:',sum(train_loss_list)/len(train_loss_list),'===')

def validation(model):
    # validation
    validation_data = np.load(os.path.join(INPUT_DIR,'valid.npy'))
    #model.load_state_dict(torch.load("./weight_"+TARGET_DIR+'_'+str(EPOCHS)+".pth"))
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    loss = 0
    validation_loss_list = []
    with torch.no_grad():
        for i, x in enumerate(validation_dataloader):
            # feed forward
            x = x.float()
            x = x.view(-1, SEQ_LEN, VEC_LEN)
            #x = x.view(-1, SEQ_LEN_sqrt, SEQ_LEN_sqrt)
            x = x.to(device)
            result,atten_weight = model(x)
            
            # calculate loss
            x = x.view(-1,SEQ_LEN,VEC_LEN)
            reconstr_loss = criterion(result, x)
            validation_loss_list.append(reconstr_loss.item())
            
            # print progress
            if(i % LOG_INTERVAL == 0):
                print('{}/{}, loss = {}'.format(i,len(validation_data)//BATCH_SIZE,reconstr_loss))
        print('=== Validation Avg. Loss:',sum(validation_loss_list)/len(validation_loss_list),'===')
# test attack data
def test_attack_data(model):
    attack_data = np.load(os.path.join(INPUT_DIR,'attack.npy'))
    #model.load_state_dict(torch.load("./weight_"+TARGET_DIR+'_'+str(EPOCHS)+".pth"))
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
            #x = x.view(-1, SEQ_LEN_sqrt, SEQ_LEN_sqrt)
            x = x.to(device)
            result,atten_weight = model(x)
            
            # calculate loss
            x = x.view(-1,SEQ_LEN,VEC_LEN)
            reconstr_loss = criterion(result, x)
            attack_loss_list.append(reconstr_loss.item())
            
        print('=== Attack avg. loss = {} ==='.format(sum(attack_loss_list)/len(attack_loss_list)))


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
    model = CMAE(seq_len=SEQ_LEN,hidden_size=HIDDEN_SIZE,mem_dim=MEM_DIM,shrink_thres=SHRINK_THRESHOLD).to(device)
    criterion = nn.MSELoss()
    entropy_loss_func = EntropyLossEncap().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # train
    train(model)

    validation(model)

    test_attack_data(model)



        
