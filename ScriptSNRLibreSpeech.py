import os
import torchaudio
import torch
from torchmetrics import SignalNoiseRatio
import matplotlib.pyplot as plt
from torch.nn import Module, Linear, Sigmoid, LSTM, BCELoss, MSELoss
from torch.optim import Adam
import torch.nn.functional as F
from pytorch_model_summary import summary
from tqdm import tqdm
import numpy as np
import random
import speechbrain as sb
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper
import pickle

REFERENCE_CHANNEL = 0
SAME_LENGTH = True
THR_S = 0.5 # Threshold for speech IRM
THR_N = 0.5
N_PATH = "/project/data_asr/CHiME5/data/librenoise/free-sound/"#"/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/noise/free-sound/"
S_PATH = "/project/data_asr/CHiME5/data/librenoise/dev/dev-clean/"#"/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/LibriSpeech/dev-clean/"
PICKLE_SAVE_PATH = '/project/data_asr/CHiME5/data/librenoise/models/params.pkl'
MODEL_SAVE_PATH = '/project/data_asr/CHiME5/data/librenoise/models/modelLibre'

def load_noise(N_PATH=N_PATH):
    noise = []
    for file in  os.listdir(N_PATH):
        if file[-4:] == ".wav":
            sound, _ = torchaudio.load(N_PATH+file)
            noise.append(sound)
    return noise


def load_speech(S_PATH=S_PATH):
    speech = []
    for folder in  os.listdir(S_PATH):
        if os.path.isdir(S_PATH+folder):
            for subfolder in os.listdir(S_PATH+folder):
                if os.path.isdir(S_PATH+folder+"/"+subfolder):
                    for file in os.listdir(S_PATH+folder+"/"+subfolder):
                        if file[-5:] == ".flac":
                            sound, _ = torchaudio.load(S_PATH+folder+"/"+subfolder+"/"+file)
                            if SAME_LENGTH:
                                try:
                                    sound = torch.narrow(sound,1,0,50000)# Narrow to 50000
                                except Exception:
                                    # add zeros to make sound 50000 long
                                    len_sound = sound.shape[1]
                                    add_zeros = 50000 - len_sound
                                    add_zeros = torch.zeros(add_zeros).reshape(1,-1)
                                    sound = torch.concat([sound,add_zeros],dim=1)
                            speech.append(sound)
    return speech

def add_noise_to_speech(speech, noise, ratio1: float, ratio2: float):
    X = []
    X2 = []
    newNoise = []
    for sample in speech:
        len_speech = sample.shape[1]
        sample_noise = random.choice(noise)
        while sample_noise.shape[1]<len_speech:
            sample_noise = torch.concat([sample_noise,sample_noise],dim=1)# Repeat to ensure noise is longer than speech
        sample_noise = torch.narrow(sample_noise,1,0,len_speech)# Shorten noise to same length as speech
        x = torch.add(sample,sample_noise*ratio1)# Same Ratio 1:1
        x2 = torch.add(sample,sample_noise*ratio2)# Same Ratio 1:1
        sample_noise = torch.narrow(sample_noise,1,0,len_speech)
        X.append(x)
        X2.append(x)
        newNoise.append(sample_noise)
    return X, X2, newNoise    

def prep_xij(trainX,i,j):
    real_part = trainX[i][j].real
    imag_part = trainX[i][j].imag
    return torch.cat((real_part.unsqueeze(2),imag_part.unsqueeze(2)),2)

speech = load_speech()
noise = load_noise()
#noise = noise[:5]
mix1, mix2, noise = add_noise_to_speech(speech, noise, 0.1, 0.3)

print(len(speech), speech[0].shape)
print(len(noise), noise[0].shape)
print(len(mix1), mix2[0].shape)

# STFT
N_FFT = 1024
N_HOP = 256

stft = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=N_HOP,
    power=None,
)
istft = torchaudio.transforms.InverseSpectrogram(n_fft=N_FFT, hop_length=N_HOP)

stfts_mix = []
for n in range(0,len(mix1)):
    x_new = torch.concat([stft(mix1[n]),stft(mix2[n])],dim=0)
    stfts_mix.append(x_new)

    
stfts_clean = []
for y in speech:
    y_new = stft(y)
    y_new = y_new.reshape(513,-1)
    stfts_clean.append(y_new)

stfts_noise = []
i = 0
for n in noise:
    try:
        n_new = stft(n)
        stfts_noise.append(n_new.reshape(513,-1))
    except Exception: #sometimes noises are very short, e.g. noise[697]
        continue

print(stfts_clean[0].shape)
print(stfts_mix[0][0].shape)
print(stfts_noise[0].shape)

def get_irms(stft_clean, stft_noise):
    mag_clean = stft_clean.abs() ** 2
    mag_noise = stft_noise.abs() ** 2
    irm_speech = mag_clean / (mag_clean + mag_noise)
    irm_noise = mag_noise / (mag_clean + mag_noise)
    return irm_speech[REFERENCE_CHANNEL], irm_noise[REFERENCE_CHANNEL]

Y = []
for n in range(0,len(noise)):
    irm_speech, irm_noise = get_irms(stfts_clean[n].unsqueeze(0), stfts_noise[n].unsqueeze(0))
    irm_speech = (irm_speech>THR_S).float()
    irm_noise = (irm_noise>THR_N).float()
    Y.append(torch.cat((irm_speech.unsqueeze(0),irm_noise.unsqueeze(0)),0))

X_h = []
stfts_mix_s = torch.stack(stfts_mix)

for i in range(0, len(stfts_mix)):
    for j in range (0,1):
        X_h.append(torch.cat((stfts_mix_s[i][j].real.unsqueeze(0),stfts_mix_s[i][j].imag.unsqueeze(0)),0))

speech_s = torch.stack(speech)
noise_s = torch.stack(noise)
mix1_s = torch.stack(mix1)
Y = torch.stack(Y)
X = torch.stack(X_h)
Y = speech_s # NEW

# MASK NET
HIDDEN_SIZE=1024 # 1024 (128 is too litte, just learns all 0 or 1)
SAMPLE_RATE = 16000
INPUT_CHANNEL = 2 # Always two -> Real and Imaginary part 

class MaskNet(Module):
    def __init__(self,noise=False):
        super(MaskNet, self).__init__()
        # First subnet for speech prediction
        self.noise = noise
        self.lstm = LSTM(input_size=INPUT_CHANNEL, hidden_size=HIDDEN_SIZE, num_layers=2, bidirectional=True)
        self.fc = Linear(in_features=HIDDEN_SIZE*2 ,out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self,x):
        # Speech prediction
        x = x.reshape(513,-1,2)
        self.lstm.flatten_parameters()
        y, (h_n, c_n) = self.lstm(x)
        y = self.fc(y)
        speech_pred = self.sigmoid(y)
        return speech_pred#, noise_pred

print(summary(MaskNet(),torch.zeros((2, 196, 513))))

EPOCHS = 30
BATCH_SIZE = 1
REFERENCE_CHANNEL = 0
INIT_LR = 0.01

CUDA = True # if torch.cuda.is_available()
device =  torch.device("cuda:2") if torch.cuda.is_available() else torch.device('cpu')
print("Mounted on:", device)

#lossBCE = MSELoss()
#lossSiSNR = get_si_snr_with_pitwrapper([1,],[])
lossBCE = SignalNoiseRatio().to(device)

model = MaskNet().to(device)
model= torch.nn.DataParallel(model,device_ids=[2,3])
opt = Adam(model.parameters(), lr=INIT_LR)

H = {
    "train_loss":[],
    "train_acc":[],
    "val_loss":[],
    "val_acc":[]
}

def check_accuracy_training(speech_pred, y_s):
    return float(torch.add(speech_pred,-y_s))

def check_accuracy_validation(model):
    example_nr = int(np.random.random()*(len(speech)-len(trainX))+len(trainX))
    model.eval()
    pred=maskToWave(model(X[example_nr]).reshape(513,-1),example_nr)
    val_loss = lossBCE(pred,Y[example_nr])
    model.train()
    return val_loss

istft = torchaudio.transforms.InverseSpectrogram(n_fft=N_FFT, hop_length=N_HOP).to(device)

def maskToWave(speech_pred,i):
        mix = stfts_mix[i].to(device)
        noise_pred = torch.ones([513,mix.shape[2]]).to(device)-speech_pred
        psd_transform = torchaudio.transforms.PSD()
        psd_speech = psd_transform(mix, speech_pred)
        psd_noise = psd_transform(mix, noise_pred)
        mvdr_transform = torchaudio.transforms.SoudenMVDR()
        stft_souden = mvdr_transform(mix, psd_speech, psd_noise, reference_channel=REFERENCE_CHANNEL)
        waveform_souden = istft(stft_souden, length=50000)#X[i].shape[-1])
        return waveform_souden.unsqueeze(0)

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

    X = X.to(device)
    Y = Y.to(device)
    trainX = X[:2000]
    trainY = Y
    for i in tqdm(range(0,len(trainX))): # Iterate over Training Examples
        (x, y) = (trainX[i],trainY[i][0].unsqueeze(0))
        speech_pred=maskToWave(model(x).reshape(513,-1),i)
        loss = lossBCE(speech_pred,y)
        # zero out the gradients, perform the backpropagation step, and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        #H["train_acc"].append(check_accuracy_training(speech_pred,y))
        H["train_loss"].append(float(loss))
        if i % 10 == 0:
            val_loss = check_accuracy_validation(model)
        #    H["val_acc"].append(val_acc)
            H["val_loss"].append(float(val_loss))
        # if i % 100 == 0:
        #     if i == 0:
        #         continue
        #     #print("Average Training Accuracy at Iteration",str(i),":",np.mean(np.array(H["train_acc"])))
        #     # print("Total Training Loss at Iteration",str(i),":",np.sum(np.array(H["train_loss"])))
        #     # # print("Average Validation Accuracy at Iteration",str(i),":",np.mean(np.array(H["val_acc"])))
        #     # print("Total Validation Loss at Iteration",str(i),":",np.sum(np.array(H["val_loss"])))
        #     # # Reset H
        #     # # H =S {
        #     #     "train_loss":[],
        #     #     "val_acc":[]
        #     # }
    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH + "epoch"+ str(epoch+1) + ".pt")

torch.save(model.state_dict(), MODEL_SAVE_PATH + "final" + ".pt")
with open(PICKLE_SAVE_PATH, 'wb') as f:
    pickle.dump(H, f)