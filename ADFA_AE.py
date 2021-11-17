import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sklearn
from sklearn.model_selection import train_test_split
from preprocess import Preprocess # import from "./preprocess.py"
from model import AE # import from "./model.py"

# Global Variables
INPUT_DIR = "/home/vincent/Desktop/research/ADFA-LD"
NEED_PREPROCESS = False
SEQ_LEN = 20 # n-gram length
TOTAL_SYSCALL_NUM = 334
EPOCHS = 20 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 128 # batch size for training
HIDDEN_SIZE = 256 # encoder's 1st lstm layer hidden size 
DROP_OUT = 0.0
VEC_LEN = 1 # length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
LOG_INTERVAL = 100 # log interval of printing message

def preprocess_data():
    # Preprocess data (if needed)
    prep = Preprocess(seq_len=SEQ_LEN,total_syscall_num=TOTAL_SYSCALL_NUM)
    prep.read_files(INPUT_DIR)
    prep.output_files(INPUT_DIR)

def train(model):
    # training
    train_data = np.load(os.path.join(INPUT_DIR,'train.npy'))
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)
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
            x = x.view(-1,SEQ_LEN)
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
    print(sum(train_loss_list)/len(train_loss_list))
    torch.save(model.state_dict(), "./weight.pth")

def validation(model):
    # validation
    validation_data = np.load(os.path.join(INPUT_DIR,'validation.npy'))
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE,shuffle=True)
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
            x = x.view(-1,SEQ_LEN)
            validation_loss = criterion(result, x)
            validation_loss_list.append(validation_loss.item())
            
            # print progress
            if(i % LOG_INTERVAL == 0):
                print('{}/{}, loss = {}'.format(i,len(validation_data)//BATCH_SIZE,validation_loss))
        print(sum(validation_loss_list)/len(validation_loss_list))
# test attack data
def test_attack_data(model,attack_type='Adduser'):
    attack_data = np.load(os.path.join(INPUT_DIR,attack_type+'.npy'))
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()
    attack_dataloader = DataLoader(attack_data, batch_size=BATCH_SIZE,shuffle=True)
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
            x = x.view(-1,SEQ_LEN)
            attack_loss = criterion(result, x)
            attack_loss_list.append(attack_loss.item())
            
            # print progress
            if(i % LOG_INTERVAL == 0):
                print('{}/{}, loss = {}'.format(i,len(attack_data)//BATCH_SIZE,attack_loss))
        print('=== Attack type = {}, Avg loss = {:.10f} ==='.format(attack_type,sum(attack_loss_list)/len(attack_loss_list)))

if __name__ == '__main__':  
    # Check if using GPU
    print("Is using GPU?",torch.cuda.is_available())
    device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:",device)
    print("Currently using GPU:",torch.cuda.get_device_name(0))

    # model setting
    ae_model = AE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae_model.parameters(), lr=LR)

    # preprocess data
    if(NEED_PREPROCESS):
        preprocess_data()

    # train
    train(ae_model)

    # validation
    #validation(ae_model)

    # test attack data
    # attack_list = ['Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
    # for attack_type in attack_list:
    #     test_attack_data(ae_model,attack_type=attack_type)