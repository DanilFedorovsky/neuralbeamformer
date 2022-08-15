import os
import torchaudio
import torch
from torch.nn import Module, Linear, Sigmoid, LSTM, MSELoss, Parameter
from torch.optim import Adam
from pytorch_model_summary import summary
from tqdm import tqdm

N_PATH = "/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/noise/free-sound/"
S_PATH = "/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/LibriSpeech/dev-clean/"
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


def add_noise_to_speech(speech, noise):
    X = []
    for sample in speech:
        len_speech = sample.shape[1]
        sample_noise = torch.concat([noise[6],noise[6],noise[6],noise[6]],dim=1)# Repeat to ensure noise is longer than speech
        sample_noise = torch.narrow(sample_noise,1,0,len_speech)# Shorten noise to same length as speech
        x = torch.add(sample,sample_noise)# Same Ratio 1:1
        X.append(x)
    return X    

def get_one_noise(speech, noise):# Make Noise 6 same length as respective speech
    X = []
    for sample in speech:
        len_speech = sample.shape[1]
        sample_noise = torch.concat([noise[6],noise[6],noise[6],noise[6]],dim=1)# Repeat noise to ensure noise is longer than speech
        sample_noise = torch.narrow(sample_noise,1,0,len_speech)# Shorten noise to same length as speech
        x = sample_noise# Same Ratio 1:1
        X.append(x)
    return X

speech = load_speech()
noise = load_noise()
X = add_noise_to_speech(speech, noise)
noise = get_one_noise(speech,noise)

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
for x in X:
    # Add second channel with double Amplitude
    x2 = x*2
    x_new = torch.concat([stft(x),stft(x2)],dim=0) # mel_stft
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

trainX = stfts_mix
trainY_speech = stfts_clean
trainY_noise = stfts_noise

def prep_xij(trainX,i,j):
    real_part = trainX[i][j].real
    imag_part = trainX[i][j].imag
    return torch.cat((real_part.unsqueeze(2),imag_part.unsqueeze(2)),2)

print(stfts_clean[0].shape)
print(stfts_mix[0][0].shape)
print(stfts_noise[0].shape)

# MASK NET
HIDDEN_SIZE=128 # 128
SAMPLE_RATE = 16000
INPUT_CHANNEL = 2 # Always two -> Real and Imaginary part 
REFERENCE_CHANNEL = 0

class MultiTaskLossWrapper(Module):
    # https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = Parameter(torch.zeros((task_num)))

    def forward(self, preds, y0, y1, y2, y3):

        mse = MSELoss()

        loss0 = mse(preds[0][0],y0)
        loss1 = mse(preds[0][1],y1)
        loss2 = mse(preds[1][0],y2)
        loss3 = mse(preds[1][1],y3)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]

        precision3 = torch.exp(-self.log_vars[3])
        loss3 = precision3*loss3 + self.log_vars[3]
        
        print(precision0,precision1,precision2,precision3)
        return loss0+loss1+loss2+loss3


class MaskNet(Module):
    def __init__(self,noise=False):
        super(MaskNet, self).__init__()
        # First subnet for speech prediction
        self.noise = noise
        self.lstm = LSTM(input_size=INPUT_CHANNEL, hidden_size=HIDDEN_SIZE, num_layers=2, bidirectional=True)
        self.fc = Linear(in_features=HIDDEN_SIZE*2 ,out_features=2)
        self.sigmoid = Sigmoid()
        # Second subnet for noise prediction
        self.noise2 = noise
        self.lstm2 = LSTM(input_size=INPUT_CHANNEL, hidden_size=HIDDEN_SIZE, num_layers=2, bidirectional=True)
        self.fc2 = Linear(in_features=HIDDEN_SIZE*2 ,out_features=2)
        self.sigmoid2 = Sigmoid()

    def forward(self,x):
        # Speech prediction
        y, (h_n, c_n) = self.lstm(x)
        y = self.fc(y).type(torch.double)
        speech_pred = self.sigmoid(y)
        speech_pred = speech_pred.reshape(2,513,-1)

        # Noise prediction
        z, (h_n, c_n) = self.lstm2(x)
        z = self.fc2(z).type(torch.double)
        noise_pred = self.sigmoid2(z)
        noise_pred = noise_pred.reshape(2,513,-1)

        return speech_pred.type(torch.float32), noise_pred.type(torch.float32)

print(summary(MaskNet(),torch.zeros((513, 468, 2))))


EPOCHS = 1
NUM_CHANNEL = 2 # Number of Mic Inputs (>=2 for BF)
REFERENCE_CHANNEL = 0
INIT_LR = 0.01
BATCH_SIZE = 1#64
LEARN_LOSS_PARAMS = False
#device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = MaskNet()#.to(device)

if LEARN_LOSS_PARAMS:
    mse_wrapper = MultiTaskLossWrapper(task_num=4)
else:
    lossMSE = MSELoss() # Better Compare MEL instead MSE for both speech and noise
#model.type(torch.float)

opt = Adam(model.parameters(), lr=INIT_LR)


H = {
    "train_loss":[],
    "train_acc":[],
    "val_loss":[],
    "val_acc":[]
}

print("[INFO] training the network...")
#startTime = time.time()

for epoch in range(0, EPOCHS):
    print("Epoch:",str(epoch)+"/"+str(EPOCHS))
    # Train Mode
    model.train()
    
    # Initialize
    totalTrainLoss = 0
    totalValLoss = 0
    trainCorrect = 0
    valCorrect = 0

    trainX = stfts_mix
    trainY_speech = stfts_clean
    trainY_noise = stfts_noise

    for i in tqdm(range(0,30)):#len(trainX))): # Iterate over Training Examples
        for j in range(0,NUM_CHANNEL):# + Iterate over channels
            (x, y_s, y_n) = (prep_xij(trainX,i,j),trainY_speech[i],trainY_noise[i])
            #(x, y_s, y_n) = (torch.cat((trainX[i][0].unsqueeze(2),trainX[i][1].unsqueeze(2)),2),trainY_speech[i],trainY_noise[i])
            speech_pred, noise_pred = model(x)
            if LEARN_LOSS_PARAMS:
                loss = mse_wrapper([noise_pred,speech_pred],y_n.real,y_n.imag,y_s.real,y_s.imag)
            else:
                loss = lossMSE(noise_pred[0],y_n.real) + lossMSE(noise_pred[1],y_n.imag) + lossMSE(speech_pred[0], y_s.real) + lossMSE(speech_pred[1], y_s.imag)
            # print("noise_pred:", noise_pred[0])
            # print("y_n.real:",y_n.real)
            print(loss)
            # zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss
    PATH = "./modelsaveLibreOneNoise"
    torch.save(model.state_dict(), PATH + "epoch" + str(epoch) + ".pt")
    print(totalTrainLoss)

PATH = "./modelsaveLibreOneNoise.pt"
torch.save(model.state_dict(), PATH)