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
# Variational AutoEncoder
class VAE(nn.Module):
    def __init__(self,seq_len=20,vec_len=1,hidden_size=64,dropout=0.0):
        super(VAE,self).__init__()
        self.vec_len = vec_len # vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # encoder
        self.ec_lstm1 = nn.LSTM(input_size=vec_len,hidden_size=hidden_size,batch_first=True,dropout=dropout)
        self.ec_lstm2 = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size//2,batch_first=True,dropout=dropout)
        
        # mean and standard deviation
        self.mean_fc = nn.Linear(hidden_size//2,hidden_size//2)
        self.logv_fc = nn.Linear(hidden_size//2,hidden_size//2)
        
        # decoder
        self.dc_lstm1 = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size,batch_first=True,dropout=dropout)
        
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vec_len)
        self.tdd = TimeDistributed(self.fc,batch_first=True)

    def forward(self,x):
        # encode
        out, _ = self.ec_lstm1(x)
        out = F.relu(out)
        out, (hn,_) = self.ec_lstm2(out)
        hn = hn.view(-1,self.vec_len,self.hidden_size//2)

        # reparameter
        mean = self.mean_fc(hn)
        log_var = self.logv_fc(hn)
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mean + eps*std
        z = z.repeat(1,self.seq_len,self.vec_len)

        # decode
        out, _ = self.dc_lstm1(z)
        out = F.relu(out)

        # fully-connect
        fc_out = self.tdd(out)
        
        return fc_out, mean, log_var
# Convolution Variational AutoEncoder
class CVAE(nn.Module):
    def __init__(self,seq_len=20,vec_len=1,hidden_size=64,dropout=0.0):
        super(CVAE,self).__init__()
        self.vec_len = vec_len # vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.vec_len, out_channels=self.hidden_size, kernel_size=3,padding=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size//2, kernel_size=3,padding=1),
            nn.BatchNorm2d(self.hidden_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.hidden_size//2, out_channels=self.hidden_size//4, kernel_size=3,padding=1),
            nn.BatchNorm2d(self.hidden_size//4),
            nn.LeakyReLU(0.2, inplace=True)
        )


        # mean and standard deviation
        self.mean_fc = nn.Linear(hidden_size//4,hidden_size//4)
        self.logv_fc = nn.Linear(hidden_size//4,hidden_size//4)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.hidden_size//4, out_channels=self.hidden_size//2, kernel_size=3,padding=1),
            nn.BatchNorm2d(self.hidden_size//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=self.hidden_size//2, out_channels=self.hidden_size, kernel_size=3,padding=1),
        )
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vec_len)
        self.tdd = TimeDistributed(self.fc,batch_first=True)

    def forward(self,x):
        # encode
        x = x.permute(0,2,1)
        x = x.unsqueeze(3)
        x = self.encoder(x)
        x = x.squeeze(3)
        x = x.permute(0,2,1)

        # reparameter
        mean = self.mean_fc(x)
        log_var = self.logv_fc(x)
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mean + eps*std

        # decode
        z = z.permute(0,2,1)
        z = z.unsqueeze(3)
        out = self.decoder(z)
        out = out.squeeze(3)
        out = out.permute(0,2,1)

        # fully-connect
        fc_out = torch.sigmoid(self.tdd(out)) 
        #fc_out = self.tdd(out) 
        
        return fc_out, mean, log_var

# Memory-augmented AutoEncoder
class MemAE(nn.Module):
    def __init__(self,seq_len=20,vec_len=1,hidden_size=256,dropout=0.0,shrink_thres=0.0025,mem_dim=256,num_layers=1):
        super(MemAE,self).__init__()
        self.vec_len = vec_len # vec_len: length of syscall representation vector, e.g., read: 0 (after embedding might be read: [0.1,0.03,0.2])
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        # encoder
        self.ec_lstm1 = nn.LSTM(input_size=vec_len,hidden_size=hidden_size,batch_first=True,dropout=dropout,num_layers=num_layers)
        self.ec_lstm2 = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size//2,batch_first=True,dropout=dropout,num_layers=num_layers)
        self.ec_lstm3 = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size//4,batch_first=True,dropout=dropout,num_layers=num_layers)
        
        # Memory module
        self.mem_module = MemModule(mem_dim=mem_dim, fea_dim=hidden_size//4, shrink_thres=shrink_thres)

        # decoder
        self.dc_lstm1 = nn.LSTM(input_size=hidden_size//4,hidden_size=hidden_size//2,batch_first=True,dropout=dropout,num_layers=num_layers)
        self.dc_lstm2 = nn.LSTM(input_size=hidden_size//2,hidden_size=hidden_size,batch_first=True,dropout=dropout,num_layers=num_layers)
        
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vec_len)
        self.tdd = TimeDistributed(self.fc,batch_first=True)
    def forward(self,x):
        # encode
        out, _ = self.ec_lstm1(x)
        out = F.relu(out)
        out, _ = self.ec_lstm2(out)
        out = F.relu(out)
        latent_vec, (hn,_)  = self.ec_lstm3(out)
        
        # memory module
        latent_vec = latent_vec.permute(0,2,1)
        z, atten_weight = self.mem_module(latent_vec)
        z = z.permute(0,2,1)
        atten_weight = atten_weight.permute(0,2,1)

        # decode
        out, _ = self.dc_lstm1(z)
        out = F.relu(out)
        out, _ = self.dc_lstm2(out)
        dc_out = F.relu(out)

        # fully-connect
        fc_out = self.tdd(dc_out)
        
        return fc_out, atten_weight

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