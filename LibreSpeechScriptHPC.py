import os
import torchaudio
import torch
import matplotlib.pyplot as plt
from torch.nn import Module, Linear, Sigmoid, LSTM, BCELoss
from torch.optim import Adam
import torch.nn.functional as F
from pytorch_model_summary import summary
from tqdm import tqdm
import numpy as np
import random

REFERENCE_CHANNEL = 0
SAME_LENGTH = False
THR_S = 0.5 # Threshold for speech IRM
THR_N = 0.5
N_PATH = "../free-sound/"#"/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/noise/free-sound/"
S_PATH = "../dev-clean/"#"/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/LibriSpeech/dev-clean/"
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

# STFT
N_FFT = 1024
N_HOP = 256

stft = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=N_HOP,
    power=None,
)
istft = torchaudio.transforms.InverseSpectrogram(n_fft=N_FFT, hop_length=N_HOP)


speech = load_speech()
noise = load_noise()
X, X2, noise = add_noise_to_speech(speech, noise, 0.2, 0.5)

stfts_mix = []
for n in range(0,len(X)):
    x_new = torch.concat([stft(X[n]),stft(X2[n])],dim=0)
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

trainX = stfts_mix

def get_irms(stft_clean, stft_noise):
    mag_clean = stft_clean.abs() ** 2
    mag_noise = stft_noise.abs() ** 2
    irm_speech = mag_clean / (mag_clean + mag_noise)
    irm_noise = mag_noise / (mag_clean + mag_noise)
    return irm_speech[REFERENCE_CHANNEL], irm_noise[REFERENCE_CHANNEL]

trainY = []
for n in range(0,len(noise)):
    irm_speech, irm_noise = get_irms(stfts_clean[n].unsqueeze(0), stfts_noise[n].unsqueeze(0))
    irm_speech = (irm_speech>THR_S).float()
    irm_noise = (irm_noise>THR_N).float()
    trainY.append(torch.cat((irm_speech.unsqueeze(0),irm_noise.unsqueeze(0)),0))

# MASK NET
HIDDEN_SIZE=256 # 128
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
        y, (h_n, c_n) = self.lstm(x)
        y = self.fc(y)
        speech_pred = self.sigmoid(y)
        return speech_pred.reshape(513,-1)#, noise_pred

print(summary(MaskNet(),torch.zeros((513, 196, 2))))

EPOCHS = 1
NUM_CHANNEL = 2 # Number of Mic Inputs (>=2 for BF)
REFERENCE_CHANNEL = 0
INIT_LR = 0.01
BATCH_SIZE = 1#64
LEARN_LOSS_PARAMS = False
device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = MaskNet()#.to(device)

lossBCE = BCELoss()

opt = Adam(model.parameters(), lr=INIT_LR)


H = {
    "train_loss":[],
    "train_acc":[],
    "val_loss":[],
    "val_acc":[]
}

def check_accuracy_training(speech_pred, y_s):
    speech_pred = (speech_pred>0.15).float()
    return float(torch.sum((speech_pred == y_s).float())/torch.sum(torch.ones(513,speech_pred.shape[1])))

def check_accuracy_validation(model):
    example_nr = int(np.random.random()*(len(speech)-len(trainX))+len(trainX))
    model.eval()
    pred = model(prep_xij(stfts_mix,example_nr,0))
    val_loss = lossBCE(pred,trainY[example_nr][0])
    pred = (pred>0.15).float()
    model.train()
    return float(torch.sum((pred == trainY[example_nr][0]).float())/torch.sum(torch.ones(513,stfts_mix[example_nr].shape[2]))),val_loss

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

    trainX = stfts_mix[:1000]#stfts_mix[:50]

    for i in tqdm(range(0,len(trainX))): # Iterate over Training Examples
        for j in range(0,NUM_CHANNEL):# + Iterate over channels
            (x, y_s, y_n) = (prep_xij(trainX,i,j),trainY[i][0],trainY[i][1])
            speech_pred=model(x)#, noise_pred = model(x)
            if LEARN_LOSS_PARAMS:
                pass
                #loss = mse_wrapper([speech_pred_real, speech_pred_imag, noise_pred_real, noise_pred_imag],y_n.real,y_n.imag,y_s.real,y_s.imag)
            else:
                #noise_pred = torch.ones([196])-speech_pred
                loss = lossBCE(speech_pred,y_s) #+ lossMSE(noise_pred,y_n)
            # zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            H["train_acc"].append(check_accuracy_training(speech_pred,y_s))
            H["train_loss"].append(float(loss))
            if i % 10 == 0:
                val_acc, val_loss = check_accuracy_validation(model)
                H["val_acc"].append(val_acc)
                H["val_loss"].append(float(val_loss))
    # Print results of Epoch        
    print("Average Training Accuracy in Epoch",str(epoch+1),":",np.mean(np.array(H["train_acc"])))
    print("Total Training Loss in Epoch",str(epoch+1),":",np.sum(np.array(H["train_loss"])))
    print("Average Validation Accuracy in Epoch",str(epoch+1),":",np.mean(np.array(H["val_acc"])))
    print("Total Validation Loss in Epoch",str(epoch+1),":",np.sum(np.array(H["val_loss"])))
    # Save Model after Epoch        
    PATH = "./modelLibre"
    torch.save(model.state_dict(), PATH + ".pt") #+ "model_epoch" + str(epoch+1)

def evaluateSiSNR(wave, i):
    def si_snr(estimate, reference, epsilon=1e-8):
        estimate = estimate - estimate.mean()
        reference = reference - reference.mean()
        reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
        mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
        scale = mix_pow / (reference_pow + epsilon)

        reference = scale * reference
        error = estimate - reference

        reference_pow = reference.pow(2)
        error_pow = error.pow(2)

        reference_pow = reference_pow.mean(axis=1)
        error_pow = error_pow.mean(axis=1)

        si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
        return si_snr.item()
    score = si_snr(wave, speech[i])
    print(f"Si-SNR score: {score}") 
    return score

def maskToWave(speech_pred,noise_pred,mix,i):
    model.eval()
    psd_transform = torchaudio.transforms.PSD()
    psd_speech = psd_transform(mix[i], speech_pred)
    psd_noise = psd_transform(mix[i], noise_pred)
    mvdr_transform = torchaudio.transforms.SoudenMVDR()
    stft_souden = mvdr_transform(mix[i], psd_speech, psd_noise, reference_channel=REFERENCE_CHANNEL)
    waveform_souden = istft(stft_souden, length=len(X[i][0]))#X[i].shape[-1])
    return waveform_souden.reshape(-1)

def save_sample(i,wave,sample_rate=SAMPLE_RATE):
    model.eval()
    wave = wave
    torchaudio.save("sampleall_output.wav",wave.reshape(1,-1),sample_rate)
    torchaudio.save("sampleall_reference.wav", speech[i].reshape(1,-1),sample_rate)
    torchaudio.save("sampleall_input.wav", X[i].reshape(1,-1),sample_rate)


sample_nr = 1001
speech_pred = (model(prep_xij(stfts_mix,sample_nr,0))>0.15).float()
noise_pred = torch.ones([513,speech_pred.shape[1]])-speech_pred
wave = maskToWave(speech_pred,noise_pred,stfts_mix,sample_nr)
save_sample(sample_nr, wave=wave)
score = evaluateSiSNR(wave,sample_nr)#Should be ~16
print(score)