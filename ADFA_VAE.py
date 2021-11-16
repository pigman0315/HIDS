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

# Hyperparameters
EPOCHS = 20 # epoch
LR = 0.0001  # learning rate
BATCH_SIZE = 100 # batch size for training
HIDDEN_SIZE = 256
DROP_OUT = 0.0
VEC_LEN = 1

# training setting
ae_model = AE().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(ae_model.parameters(), lr=LR)
dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)

# training
loss_list = []
for epoch in range(EPOCHS):
    loss = 0
    for i, x in enumerate(dataloader):
        # feed forward
        x = x.float()
        x = x.view(-1, SEQ_LEN, VEC_LEN)
        x = x.to(device)
        result = ae_model(x)
        
        # backpropagation
        x = x.view(-1,SEQ_LEN)
        loss = criterion(result, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print progress
        if(i % 100 == 0):
            print('Epoch {}({}/{}), loss = {}'.format(epoch,i,len(train_data)//100,loss))
        loss_list.append(loss)
    print('epoch: {}, loss: {}'.format(epoch,loss))
