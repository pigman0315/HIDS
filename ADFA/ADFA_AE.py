import os
import numpy as np
import time
import statistics as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sklearn
from sklearn.model_selection import train_test_split
from ADFA_preprocess import Preprocess # import from "./preprocess.py"
from ADFA_model import AE,CAE # import from "./model.py"

# Global Variables
INPUT_DIR = '../../ADFA-LD'
NEED_PREPROCESS = False
NEED_TRAIN = False
SEQ_LEN = 20 # n-gram length
TOTAL_SYSCALL_NUM = 340
EPOCHS = 10 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 128 # batch size for training
HIDDEN_SIZE = 256 # encoder's 1st lstm layer hidden size 
DROPOUT = 0.0
VEC_LEN = 1 # length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
LOG_INTERVAL = 1000 # log interval of printing message

def preprocess_data():
    # Preprocess data (if needed)
    prep = Preprocess(seq_len=SEQ_LEN,total_syscall_num=TOTAL_SYSCALL_NUM)
    prep.read_files(INPUT_DIR)
    prep.output_files(INPUT_DIR)

def train(model):
    # training
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
    print('=== Train Avg. Loss: {}, std: {} ==='.format(sum(train_loss_list)/len(train_loss_list),st.pstdev(train_loss_list)))

def validation(model):
    # validation
    validation_data = np.load(os.path.join(INPUT_DIR,'validation.npy'))
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
            validation_loss = validation_loss.view(-1,SEQ_LEN).to('cpu')
            for vl in validation_loss:
                vl = vl.tolist()
                validation_loss_list.append(sum(vl))
            
            # print progress
            if(i % LOG_INTERVAL == 0):
               print('{}/{}'.format(i,len(validation_data)//BATCH_SIZE))
        print('=== Validation Avg. Loss: {}, std: {} ==='.format(sum(validation_loss_list)/len(validation_loss_list),st.pstdev(validation_loss_list)))
# test attack data
def test_attack_data(model,attack_type):
    attack_data = np.load(os.path.join(INPUT_DIR,attack_type+'.npy'))
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
            attack_loss = attack_loss.view(-1,SEQ_LEN).to('cpu')
            for al in attack_loss:
                al = al.tolist()
                attack_loss_list.append(sum(al))
            
        print('=== Attack type = {}, Avg loss = {}, std = {} ==='.format(attack_type,sum(attack_loss_list)/len(attack_loss_list),st.pstdev(attack_loss_list)))

if __name__ == '__main__':  
    # Check if using GPU
    print("Is using GPU?",torch.cuda.is_available())
    device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:",device)
    print("Currently using GPU:",torch.cuda.get_device_name(0))

    # model setting
    model = CAE(seq_len=SEQ_LEN,hidden_size=HIDDEN_SIZE,dropout=DROPOUT).to(device)
    #model = AE(seq_len=SEQ_LEN,hidden_size=HIDDEN_SIZE,dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if(NEED_TRAIN == True):
        criterion = nn.MSELoss()

        # preprocess data
        if(NEED_PREPROCESS):
            preprocess_data()

        start = time.time()
        # train
        train(model)
        end = time.time()
        print('Cost time: {} mins {} secs'.format((end-start)/60,(end-start)%60))

    elif(NEED_TRAIN == False):
        criterion = nn.MSELoss(reduction='none')

        start = time.time()
        # validation
        validation(model)

        # test attack data
        attack_list = ['Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
        for attack_type in attack_list:
            test_attack_data(model,attack_type=attack_type)
        end = time.time()
        print('Cost time: {} mins {} secs'.format((end-start)/60,(end-start)%60))
    # test model
    #ae_model(torch.randn((48,20,1)).to(device))