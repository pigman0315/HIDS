import os
import numpy as np
import statistics as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sklearn
from sklearn.model_selection import train_test_split
from ADFA_preprocess import Preprocess # import from "./preprocess.py"
from ADFA_model import MemAE,CMAE # import from "./model.py"
from memory_module import EntropyLossEncap

# Global Variables
INPUT_DIR = '../../ADFA-LD'
NEED_PREPROCESS = False
SEQ_LEN = 10 # n-gram length
SEQ_LEN_sqrt = 12
TOTAL_SYSCALL_NUM = 340
EPOCHS = 20 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 128 # batch size for training
HIDDEN_SIZE = 256 # encoder's 1st lstm layer hidden size 
DROPOUT = 0.0
NUM_LAYERS = 1
VEC_LEN = 1 # length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
LOG_INTERVAL = 1000 # log interval of printing message
ENTROPY_LOSS_WEIGHT = 0.0002
MEM_DIM = 2000
SHRINK_THRESHOLD = 2/MEM_DIM # 1/MEM_DIM ~ 3/MEM_DIM
 
def preprocess_data():
    # Preprocess data (if needed)
    prep = Preprocess(seq_len=SEQ_LEN,total_syscall_num=TOTAL_SYSCALL_NUM)
    prep.read_files(INPUT_DIR)
    prep.output_files(INPUT_DIR)

def train(model):
    # training
    #model.load_state_dict(torch.load('weight.pth'))
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
    print('=== Train Avg. Loss: {}, std: {} ==='.format(sum(train_loss_list)/len(train_loss_list),st.pstdev(train_loss_list)))

def validation(model):
    # validation
    validation_data = np.load(os.path.join(INPUT_DIR,'validation.npy'))
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
            #if(i % LOG_INTERVAL == 0):
                #print('{}/{}, loss = {}'.format(i,len(validation_data)//BATCH_SIZE,reconstr_loss))
        print('=== Validation Avg. Loss: {}, std: {} ==='.format(sum(validation_loss_list)/len(validation_loss_list),st.pstdev(validation_loss_list)))
# test attack data
def test_attack_data(model,attack_type='Adduser'):
    attack_data = np.load(os.path.join(INPUT_DIR,attack_type+'.npy'))
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    attack_dataloader = DataLoader(attack_data, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    loss = 0
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
            
        print('=== Attack type = {}, Avg loss = {}, std = {} ==='.format(attack_type,sum(attack_loss_list)/len(attack_loss_list),st.pstdev(attack_loss_list)))

if __name__ == '__main__':  
    # Check if using GPU
    print("Is using GPU?",torch.cuda.is_available())
    device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:",device)
    print("Currently using GPU:",torch.cuda.get_device_name(0))

    # model setting
    #model = MemAE(seq_len=SEQ_LEN,hidden_size=HIDDEN_SIZE,dropout=DROPOUT,mem_dim=MEM_DIM,shrink_thres=SHRINK_THRESHOLD,num_layers=NUM_LAYERS).to(device)
    model = CMAE(seq_len=SEQ_LEN,hidden_size=HIDDEN_SIZE,mem_dim=MEM_DIM,shrink_thres=SHRINK_THRESHOLD).to(device)
    criterion = nn.MSELoss().to(device)
    entropy_loss_func = EntropyLossEncap().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # preprocess data
    if(NEED_PREPROCESS):
        preprocess_data()

    # train
    train(model)

    # validation
    validation(model)

    # test attack data
    attack_list = ['Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
    for attack_type in attack_list:
        test_attack_data(model,attack_type=attack_type)
    
    # test model
    #model(torch.randn((48,20,1)).to(device))