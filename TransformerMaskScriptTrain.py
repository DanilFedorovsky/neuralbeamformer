import os
import torchaudio
import torch
from torchmetrics import SignalNoiseRatio
import matplotlib.pyplot as plt
from torch.nn import Module,Sigmoid, Linear, BCELoss, MSELoss, Conv1d, Conv2d, MaxPool2d, Transformer, LayerNorm, PReLU, Fold, ConvTranspose1d, MultiheadAttention, Dropout
from torch.optim import Adam
import torch.nn.functional as F
from pytorch_model_summary import summary
from tqdm import tqdm
import numpy as np
import random
from torchmetrics import ScaleInvariantSignalNoiseRatio
import pickle
import math

import DataLoader

X,Y,speech,noise,mix = DataLoader.data_loader()

class PositionalEncoding(Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# TRANSFORMER MASK NET
NUMBER_OF_SPEAKERS = 2
ENCODED_TIMESTEPS = int(50000/8) # 50000 is len of training data -> 50000/8 = 6250
FOLDS = int((ENCODED_TIMESTEPS/250)*2-1)
FILTERS = 256
D_TF = 1024#1024

class TransformerMaskNet(Module):
    def __init__(self,noise=False):
        super(TransformerMaskNet, self).__init__()
        # ENCODER subnet
        self.tdnn = Conv1d(in_channels=1,out_channels=256,kernel_size=16,stride=8,padding=6)

        self.lnorm = LayerNorm(normalized_shape=(ENCODED_TIMESTEPS))
        self.lin0 = Linear(in_features=256, out_features=256)

        self.pe = PositionalEncoding(d_model=196)
        #self.pe2 = PositionalEncoding(d_model=250)
        self.ln11 = LayerNorm(normalized_shape=(513,196))
        self.ln12 = LayerNorm(normalized_shape=(513,196))
        self.ln21 = LayerNorm(normalized_shape=(513,196))
        self.ln22 = LayerNorm(normalized_shape=(513,196))
        self.ln31 = LayerNorm(normalized_shape=(513,196))
        self.ln32 = LayerNorm(normalized_shape=(513,196))
        self.ln41 = LayerNorm(normalized_shape=(513,196))
        self.ln42 = LayerNorm(normalized_shape=(513,196))
        # self.ln51 = LayerNorm(normalized_shape=(FOLDS,250))
        # self.ln52 = LayerNorm(normalized_shape=(FOLDS,250))
        # self.ln61 = LayerNorm(normalized_shape=(FOLDS,250))
        # self.ln62 = LayerNorm(normalized_shape=(FOLDS,250))
        # self.ln71 = LayerNorm(normalized_shape=(FOLDS,250))
        # self.ln72 = LayerNorm(normalized_shape=(FOLDS,250))
        # self.ln81 = LayerNorm(normalized_shape=(FOLDS,250))
        # self.ln82 = LayerNorm(normalized_shape=(FOLDS,250))

        self.mha1 = MultiheadAttention(embed_dim=196,num_heads=14,dropout=0.1)
        self.mha2 = MultiheadAttention(embed_dim=196,num_heads=14,dropout=0.1)
        self.mha3 = MultiheadAttention(embed_dim=196,num_heads=14,dropout=0.1)
        self.mha4 = MultiheadAttention(embed_dim=196,num_heads=14,dropout=0.1)
        # self.mha5 = MultiheadAttention(embed_dim=250,num_heads=10,dropout=0.1)
        # self.mha6 = MultiheadAttention(embed_dim=250,num_heads=10,dropout=0.1)
        # self.mha7 = MultiheadAttention(embed_dim=250,num_heads=10,dropout=0.1)
        # self.mha8 = MultiheadAttention(embed_dim=250,num_heads=10,dropout=0.1)

        self.lintf1 = Linear(in_features=196,out_features=D_TF)#1024 instead of 256!
        self.lintf2 = Linear(in_features=196,out_features=D_TF)
        self.lintf3 = Linear(in_features=196,out_features=D_TF)
        self.lintf4 = Linear(in_features=196,out_features=D_TF)
        # self.lintf5 = Linear(in_features=250,out_features=D_TF)
        # self.lintf6 = Linear(in_features=250,out_features=D_TF)
        # self.lintf7 = Linear(in_features=250,out_features=D_TF)
        # self.lintf8 = Linear(in_features=250,out_features=D_TF)
        self.lintf12 = Linear(in_features=D_TF,out_features=196)
        self.lintf22 = Linear(in_features=D_TF,out_features=196)
        self.lintf32 = Linear(in_features=D_TF,out_features=196)
        self.lintf42 = Linear(in_features=D_TF,out_features=196)
        # self.lintf52 = Linear(in_features=D_TF,out_features=250)
        # self.lintf62 = Linear(in_features=D_TF,out_features=250)
        # self.lintf72 = Linear(in_features=D_TF,out_features=250)
        # self.lintf82 = Linear(in_features=D_TF,out_features=250)

        self.prelu = PReLU()
        self.lin1 = Linear(in_features=196, out_features=(196))

        # self.fold = Fold(output_size=(1,ENCODED_TIMESTEPS),kernel_size=(1,250),stride=(1,125))
        self.lin2 = Linear(in_features=196, out_features=196)
        self.lin3 = Linear(in_features=NUMBER_OF_SPEAKERS, out_features=1)
        self.sigmoid = Sigmoid()
        # self.convT = ConvTranspose1d(in_channels=256,out_channels=1,kernel_size=16,stride=8, padding=4)

        #self.tf1 = Transformer(d_model = 256, nhead=8, dim_feedforward=1024)
        #self.tf2 = Transformer(d_model = 256, nhead=8, dim_feedforward=1024)
        # SEPFORMER Block
        # y = self.tf1(x,torch.rand(250,FOLDS,256))
        # x = y + x
        # y = self.tf2(x,torch.rand(250,FOLDS,256))
        # x = y + x # Residual connection

    def forward(self,x):

        x = x.reshape(NUMBER_OF_SPEAKERS,513,196)

        # Transformer 1
        y = self.pe(x)
        z = self.ln11(y)
        z, _ = self.mha1(z,z,z)
        z_2 = z+y
        z = self.ln12(z_2)
        z = self.lintf1(z)
        z = F.relu(z)
        z = self.lintf12(z)
        x = z+z_2+x
        # Transformer 2
        y = self.pe(x)
        z = self.ln21(y)
        z, _ = self.mha2(z,z,z)
        z_2 = z+y
        z = self.ln22(z_2)
        z = self.lintf2(z)
        z = F.relu(z)
        z = self.lintf22(z)
        x = z+z_2+x
        # Transformer 3
        y = self.pe(x)
        z = self.ln31(y)
        z, _ = self.mha3(z,z,z)
        z_2 = z+y
        z = self.ln32(z_2)
        z = self.lintf3(z)
        z = F.relu(z)
        z = self.lintf32(z)
        x = z+z_2+x
        # Transformer 4
        y = self.pe(x)
        z = self.ln41(y)
        z, _ = self.mha4(z,z,z)
        z_2 = z+y
        z = self.ln42(z_2)
        z = self.lintf4(z)
        z = F.relu(z)
        z = self.lintf42(z)
        x = z+z_2+x

        # # PERMUTATION
        # x = x.view(256,FOLDS,250)

        # # Transformer 5
        # y = self.pe2(x)
        # z = self.ln51(y)
        # z, _ = self.mha5(z,z,z)
        # z_2 = z+y
        # z = self.ln52(z_2)
        # z = self.lintf5(z)
        # z = F.relu(z)
        # z = self.lintf52(z)
        # x = z+z_2+x
        # # Transformer 6
        # y = self.pe2(x)
        # z = self.ln61(y)
        # z, _ = self.mha6(z,z,z)
        # z_2 = z+y
        # z = self.ln62(z_2)
        # z = self.lintf6(z)
        # z = F.relu(z)
        # z = self.lintf62(z)
        # x = z+z_2+x
        # # Transformer 7
        # y = self.pe2(x)
        # z = self.ln71(y)
        # z, _ = self.mha7(z,z,z)
        # z_2 = z+y
        # z = self.ln72(z_2)
        # z = self.lintf7(z)
        # z = F.relu(z)
        # z = self.lintf72(z)
        # x = z+z_2+x
        # # Transformer 8
        # y = self.pe2(x)
        # z = self.ln81(y)
        # z, _ = self.mha8(z,z,z)
        # z_2 = z+y
        # z = self.ln82(z_2)
        # z = self.lintf8(z)
        # z = F.relu(z)
        # z = self.lintf82(z)
        # x = z+z_2+x

        # PRELU and Linear
        x = self.prelu(x)
        x = x.view(NUMBER_OF_SPEAKERS,513,196)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = x.view(513,196,NUMBER_OF_SPEAKERS)
        x = self.lin3(x)
        x = F.relu(x)

        x = x.view(513,196)

        x = self.sigmoid(x)
        return x
    

print(summary(TransformerMaskNet(),torch.zeros((2, 513, 196))))

EPOCHS = 10
BATCH_SIZE = 1
INIT_LR = 0.00001 #0.001 is too high
PICKLE_SAVE_PATH = '/project/data_asr/CHiME5/data/librenoise/models/TFMaskparams.pkl'
MODEL_SAVE_PATH = '/project/data_asr/CHiME5/data/librenoise/models/TFMask'
TRS = 0.5

CUDA = True # if torch.cuda.is_available()
device =  torch.device("cuda:3") if torch.cuda.is_available() else torch.device('cpu')
print("Mounted on:", device)

lossBCE = BCELoss().to(device)

model = TransformerMaskNet().to(device)
model= torch.nn.DataParallel(model,device_ids=[3])
opt = Adam(model.parameters(), lr=INIT_LR)

H = {
    "train_loss":[],
    "train_acc":[],
    "val_loss":[],
    "val_acc":[]
}

def check_accuracy_training(speech_pred, y_s):
    speech_pred = (speech_pred>TRS).float()
    return float(torch.sum((speech_pred == y_s).float())/torch.sum(torch.ones(513,speech_pred.shape[1])))

def check_accuracy_validation(model):
    example_nr = int(np.random.random()*(len(speech)-len(trainX))+len(trainX))
    model.eval()
    pred = model(X[example_nr])
    noise_pred = torch.ones([513,196]).to(device)-pred
    val_loss = lossBCE(pred,Y[example_nr][0])+lossBCE(noise_pred,Y[example_nr][1])
    pred = (pred>TRS).float()
    model.train()
    return float(torch.sum((pred == Y[example_nr][0]).float())/torch.sum(torch.ones(513,X[example_nr].shape[2])).to(device)),val_loss

print("[INFO] training the network...")

for epoch in range(0, EPOCHS):
    print("Epoch:",str(epoch+1)+"/"+str(EPOCHS))
    # Train Mode
    model.train()
    
    # Initialize
    totalTrainLoss = 0
    totalValLoss = 0
    trainCorrect = 0
    valCorrect = 0

    trainX = X[:2000].to(device)
    trainY = Y.to(device)
    Y = trainY
    for i in tqdm(range(0,len(trainX))): # Iterate over Training Examples
        (x, y) = (trainX[i],trainY[i])
        speech_pred=model(x)
        noise_pred = torch.ones([513,196]).to(device)-speech_pred
        loss = lossBCE(speech_pred,y[0])+lossBCE(noise_pred,y[1])
        # zero out the gradients, perform the backpropagation step, and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        H["train_acc"].append(check_accuracy_training(speech_pred,y[0]))
        H["train_loss"].append(float(loss))
        if i % 10 == 0:
            val_acc, val_loss = check_accuracy_validation(model)
            H["val_acc"].append(val_acc)
            H["val_loss"].append(float(val_loss))
        if i % 100 == 0:
            if i == 0:
                continue
            print("Average Training Loss at Iteration",str(i),":",(sum(H["train_loss"][-100:]))/100)
            print("Average Validation Loss at Iteration",str(i),":",(sum(H["val_loss"][-10:]))/10)
            print("Average Training Accuracy at Iteration",str(i),":",np.mean(np.array(H["train_acc"])))
            print("Average Validation Accuracy at Iteration",str(i),":",np.mean(np.array(H["val_acc"])))
    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH + "epoch"+ str(epoch+1) + ".pt")

torch.save(model.state_dict(), MODEL_SAVE_PATH + "final" + ".pt")
with open(PICKLE_SAVE_PATH, 'wb') as f:
    pickle.dump(H, f)
