import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sklearn
from sklearn.model_selection import train_test_split
from preprocess import Preprocess # import from "./preprocess.py"
from model import AE # import from "./model.py"

# Global Variables
INPUT_DIR = "/home/vincent/Desktop/research/ADFA-LD"
NEED_PREPROCESS = False
SEQ_LEN = 20
TOTAL_SYSCALL_NUM = 334

# Check if using GPU
print("Is using GPU?",torch.cuda.is_available())
device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Device:",device)
print("Currently using GPU:",torch.cuda.get_device_name(0))

# Preprocess data (if needed)
train_data = None
if(NEED_PREPROCESS):
    prep = Preprocess(seq_len=SEQ_LEN,total_syscall_num=TOTAL_SYSCALL_NUM)
    prep.read_files(INPUT_DIR)
    prep.output_files(INPUT_DIR)
    train_data = prep.train_data
else:
    train_data = np.load(os.path.join(INPUT_DIR,'train.npy'))
    validation_data = np.load(os.path.join(INPUT_DIR,'validation.npy'))
    attack_data = np.load(os.path.join(INPUT_DIR,'Web_Shell.npy'))

# Hyperparameters
EPOCHS = 10 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 128 # batch size for training
HIDDEN_SIZE = 256
DROP_OUT = 0.0
VEC_LEN = 1 # length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
LOG_INTERVAL = 100 # log interval of printing message

# model setting
ae_model = AE().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(ae_model.parameters(), lr=LR)

# training
# train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)
# loss_list = []
# for epoch in range(EPOCHS):
#     loss = 0
#     for i, x in enumerate(train_dataloader):
#         # feed forward
#         x = x.float()
#         x = x.view(-1, SEQ_LEN, VEC_LEN)
#         x = x.to(device)
#         result = ae_model(x)
        
#         # backpropagation
#         x = x.view(-1,SEQ_LEN)
#         loss = criterion(result, x)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # print progress
#         if(i % LOG_INTERVAL == 0):
#             print('Epoch {}({}/{}), loss = {}'.format(epoch+1,i,len(train_data)//LOG_INTERVAL,loss))
#         loss_list.append(loss)
#     print('=== epoch: {}, loss: {} ==='.format(epoch,loss))
# torch.save(ae_model.state_dict(), "./weight.pth")


# validation
# ae_model.eval()
# validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE,shuffle=True)
# validation_loss = 0
# validation_loss_list = []
# with torch.no_grad():
#     for i, x in enumerate(validation_dataloader):
#         # feed forward
#         x = x.float()
#         x = x.view(-1, SEQ_LEN, VEC_LEN)
#         x = x.to(device)
#         result = ae_model(x)
        
#         # calculate loss
#         x = x.view(-1,SEQ_LEN)
#         validation_loss = criterion(result, x)
#         validation_loss_list.append(validation_loss.item())
        
#         # print progress
#         if(i % LOG_INTERVAL == 0):
#             print('{}/{}, loss = {}'.format(i,len(validation_data)//LOG_INTERVAL,validation_loss))
#     print(sum(validation_loss_list)/len(validation_loss_list))

# test attack data
ae_model.eval()
attack_dataloader = DataLoader(attack_data, batch_size=BATCH_SIZE,shuffle=True)
attack_loss = 0
attack_loss_list = []
with torch.no_grad():
    for i, x in enumerate(attack_dataloader):
        # feed forward
        x = x.float()
        x = x.view(-1, SEQ_LEN, VEC_LEN)
        x = x.to(device)
        result = ae_model(x)
        
        # calculate loss
        x = x.view(-1,SEQ_LEN)
        attack_loss = criterion(result, x)
        attack_loss_list.append(attack_loss.item())
        
        # print progress
        if(i % LOG_INTERVAL == 0):
            print('{}/{}, loss = {}'.format(i,len(attack_data)//LOG_INTERVAL,attack_loss))
    print(sum(attack_loss_list)/len(attack_loss_list))