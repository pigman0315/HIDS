import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_module import MemModule
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
# Convolutional 1D AutoEncoder
class CAE(nn.Module):
    def __init__(self,seq_len=20,vec_len=1,hidden_size=256,dropout=0.0):
        super(CAE,self).__init__()
        self.vec_len = vec_len # vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.vec_len, out_channels=self.hidden_size, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size//2, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=self.hidden_size//2, out_channels=self.hidden_size//4, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size//4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.hidden_size//4, out_channels=self.hidden_size//2, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=self.hidden_size//2, out_channels=self.hidden_size, kernel_size=3,padding=1),
        )
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vec_len)
        self.tdd = TimeDistributed(self.fc,batch_first=True)

    def forward(self,x):
        # encode
        x = x.permute(0,2,1)
        x = self.encoder(x)

        # decode
        out = self.decoder(x)
        out = out.permute(0,2,1)

        # fully-connect
        fc_out = self.tdd(out) 
        
        return fc_out
# Convolutional 1D Variational AutoEncoder
class CVAE(nn.Module):
    def __init__(self,seq_len=20,vec_len=1,hidden_size=256,dropout=0.0):
        super(CVAE,self).__init__()
        self.vec_len = vec_len # vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.vec_len, out_channels=self.hidden_size, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size//2, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=self.hidden_size//2, out_channels=self.hidden_size//4, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size//4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # mean and standard deviation
        self.mean_fc = nn.Linear(self.seq_len,self.seq_len)
        self.logv_fc = nn.Linear(self.seq_len,self.seq_len)

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.hidden_size//4, out_channels=self.hidden_size//2, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=self.hidden_size//2, out_channels=self.hidden_size, kernel_size=3,padding=1),
        )
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vec_len)
        self.tdd = TimeDistributed(self.fc,batch_first=True)

    def forward(self,x):
        # encode
        x = x.permute(0,2,1)
        x = self.encoder(x)
    
        # reparameter
        mean = self.mean_fc(x)
        log_var = self.logv_fc(x)
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mean + eps*std

        # decode
        out = self.decoder(z)
        out = out.permute(0,2,1)

        # fully-connect
        fc_out = self.tdd(out) 
        
        return fc_out, mean, log_var
# LSTM AutoEncoder
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
        hn = hn.view(-1,1,self.hidden_size//4)
        hn = hn.repeat(1,self.seq_len,1)
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

# Convolution Memory-augmented AutoEncoder
class CMAE(nn.Module):
    def __init__(self,seq_len=20,vec_len=1,hidden_size=256,dropout=0.0,shrink_thres=0.0025,mem_dim=256):
        super(CMAE,self).__init__()
        self.vec_len = vec_len # vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.vec_len, out_channels=self.hidden_size, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size//2, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=self.hidden_size//2, out_channels=self.hidden_size//4, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size//4),
            nn.LeakyReLU(0.2, inplace=True)
        )

         # Memory module
        self.mem_module = MemModule(mem_dim=mem_dim, fea_dim=hidden_size//4, shrink_thres=shrink_thres)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.hidden_size//4, out_channels=self.hidden_size//2, kernel_size=3,padding=1),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=self.hidden_size//2, out_channels=self.hidden_size, kernel_size=3,padding=1),
        )
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vec_len)
        self.tdd = TimeDistributed(self.fc,batch_first=True)

    def forward(self,x):
        # encode
        x = x.permute(0,2,1)
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)

        # memory module
        z, atten_weight = self.mem_module(x)
        #print(z.shape)

        # decode
        out = self.decoder(z)
        #print(out.shape)
        out = out.permute(0,2,1)

        # fully-connect
        #fc_out = torch.sigmoid(self.tdd(out)) 
        fc_out = self.tdd(out) 
        #print(fc_out.shape)
        
        return fc_out, atten_weight