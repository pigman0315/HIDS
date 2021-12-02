import torch
import torch.nn as nn
import torch.nn.functional as F
class TimeDistributed(nn.Module): # implementation of TimeDistributed layer in Keras
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
# Basic AutoEncoder
class AE(nn.Module):
    def __init__(self,seq_len=20,vec_len=1,hidden_size=256,dropout=0.0):
        super(AE,self).__init__()
        self.vec_len = vec_len # vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        # encoder
        self.ec_lstm1 = nn.LSTM(input_size=vec_len,hidden_size=hidden_size,batch_first=True,dropout=dropout)
        self.ec_lstm2 = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size//2,batch_first=True,dropout=dropout)
        self.ec_lstm3 = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size//4,batch_first=True,dropout=dropout)
        
        # decoder
        self.dc_lstm1 = nn.LSTM(input_size=hidden_size//4,hidden_size=hidden_size//2,batch_first=True,dropout=dropout)
        self.dc_lstm2 = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size,batch_first=True,dropout=dropout)
        
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vec_len)
        self.tdd = TimeDistributed(self.fc,batch_first=True)
    def forward(self,x):
        # encode
        out, _ = self.ec_lstm1(x)
        out = F.relu(out)
        #print(out.shape)
        out, _ = self.ec_lstm2(out)
        out = F.relu(out)
        #print(out.shape)
        latent_vec, (hn,_)  = self.ec_lstm3(out)
        #print(latent_vec.shape)
        #print(hn.shape)
        hn = hn.view(-1,self.vec_len,self.hidden_size//4)
        hn = hn.repeat(1,self.seq_len,self.vec_len)
        #print(hn.shape)
        
        # decode
        out, _ = self.dc_lstm1(hn)
        out = F.relu(out)
        #print(out.shape)
        out, _ = self.dc_lstm2(out)
        dc_out = F.relu(out)
        #print(dc_out.shape)

        # fully-connect
        fc_out = self.tdd(dc_out)
        #print(fc_out.shape)
        
        return fc_out