import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self,AE_seq_len=20,AE_vec_len=1,AE_hidden_size=256,AE_dropout=0.0):
        # AE_vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        super(AE,self).__init__()
        # encoder
        self.ec_lstm1 = nn.LSTM(input_size=AE_vec_len,hidden_size=AE_hidden_size,batch_first=True,dropout=AE_dropout)
        self.ec_lstm2 = nn.LSTM(input_size=AE_hidden_size,hidden_size=AE_hidden_size//2,batch_first=True,dropout=AE_dropout)
        self.ec_lstm3 = nn.LSTM(input_size=AE_hidden_size//2,hidden_size=AE_hidden_size//4,batch_first=True,dropout=AE_dropout)
        
        # decoder
        self.dc_lstm1 = nn.LSTM(input_size=AE_hidden_size//4,hidden_size=AE_hidden_size//4,batch_first=True,dropout=AE_dropout)
        self.dc_lstm2 = nn.LSTM(input_size=AE_hidden_size//4,hidden_size=AE_hidden_size//2,batch_first=True,dropout=AE_dropout)
        self.dc_lstm3 = nn.LSTM(input_size=AE_hidden_size//2,hidden_size=AE_hidden_size,batch_first=True,dropout=AE_dropout)
        
        # fully-connected layer
        self.fc = nn.Linear(AE_hidden_size,AE_seq_len)
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