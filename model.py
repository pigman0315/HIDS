import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic AutoEncoder
class AE(nn.Module):
    def __init__(self,seq_len=20,vec_len=1,hidden_size=256,dropout=0.0):
        # vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        super(AE,self).__init__()
        # encoder
        self.ec_lstm1 = nn.LSTM(input_size=vec_len,hidden_size=hidden_size,batch_first=True,dropout=dropout)
        self.ec_lstm2 = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size//2,batch_first=True,dropout=dropout)
        self.ec_lstm3 = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size//4,batch_first=True,dropout=dropout)
        
        # decoder
        self.dc_lstm1 = nn.LSTM(input_size=hidden_size//4,hidden_size=hidden_size//4,batch_first=True,dropout=dropout)
        self.dc_lstm2 = nn.LSTM(input_size=hidden_size//4,hidden_size=hidden_size//2,batch_first=True,dropout=dropout)
        self.dc_lstm3 = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size,batch_first=True,dropout=dropout)
        
        # fully-connected layer
        self.fc = nn.Linear(hidden_size,seq_len)
    def forward(self,x):
        # encode
        out, _ = self.ec_lstm1(x)
        out, _ = self.ec_lstm2(out)
        latent_vec, _  = self.ec_lstm3(out)
        
        # decode
        out, _ = self.dc_lstm1(latent_vec)
        out, _ = self.dc_lstm2(out)
        dc_out, _ = self.dc_lstm3(out)

        # fully-connect
        dc_out = dc_out[:,-1,:] # just want last time step hidden states
        fc_out = self.fc(dc_out)
        
        return fc_out

# Variational AutoEncoder
class VAE(nn.Module):
    def __init__(self,seq_len=20,vec_len=1,hidden_size=64,dropout=0.0):
        # vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        super(VAE,self).__init__()
        
        # encoder
        self.ec_lstm1 = nn.LSTM(input_size=vec_len,hidden_size=hidden_size,batch_first=True,dropout=dropout)
        self.ec_lstm2 = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size//2,batch_first=True,dropout=dropout)
        
        # mean and standard deviation
        self.mean_lstm = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size//2,batch_first=True,dropout=dropout)
        self.log_var_lstm = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size//2,batch_first=True,dropout=dropout)
        
        # decoder
        self.dc_lstm1 = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size//2,batch_first=True,dropout=dropout)
        self.dc_lstm2 = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size,batch_first=True,dropout=dropout)
        
        # fully-connected layer
        self.fc = nn.Linear(hidden_size,seq_len)

    def forward(self,x):
        # encode
        ec1, _ = self.ec_lstm1(x)
        ec1 = F.relu(ec1)
        h, _ = self.ec_lstm2(ec1)
        h = F.relu(h)
        mean, _ = self.mean_lstm(h)
        log_var, _ = self.log_var_lstm(h)

        # reparameter
        std = torch.exp(log_var)**0.5
        eps = torch.randn_like(std)
        z = mean + eps*std
        z = F.relu(z)

        # decode
        out, _ = self.dc_lstm1(z)
        dc_out, _ = self.dc_lstm2(out)

        # fully-connect
        dc_out = dc_out[:,-1,:] # just want last time step hidden states
        fc_out = torch.sigmoid(self.fc(dc_out))
        
        return fc_out, mean, log_var