import os
import torchaudio
import torch

def data_loader(y_mask=True, n_noise = -1):
    REFERENCE_CHANNEL = 0
    SAME_LENGTH = True
    THR_S = 0.5 # Threshold for speech IRM
    THR_N = 0.5
    N_PATH = "/project/data_asr/CHiME5/data/librenoise/free-sound/"#"/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/noise/free-sound/"
    S_PATH = "/project/data_asr/CHiME5/data/librenoise/dev/dev-clean/"
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

    def add_noise_to_speech(speech, noise):
        X = []
        for sample in speech:
            len_speech = sample.shape[1]
            sample_noise = torch.concat([noise[6],noise[6],noise[6],noise[6]],dim=1)# Repeat to ensure noise is longer than speech
            sample_noise = torch.narrow(sample_noise,1,0,len_speech)# Shorten noise to same length as speech
            x = torch.add(sample,sample_noise*0.2)# Same Ratio 1:1
            X.append(x)
        return X    

    def add_noise_to_speech2(speech, noise):
        X = []
        for sample in speech:
            len_speech = sample.shape[1]
            sample_noise = torch.concat([noise[6],noise[6],noise[6],noise[6]],dim=1)# Repeat to ensure noise is longer than speech
            sample_noise = torch.narrow(sample_noise,1,0,len_speech)# Shorten noise to same length as speech
            x = torch.add(sample,sample_noise*0.5)# Same Ratio 1:1
            X.append(x)
        return X 

    def get_one_noise(speech, noise):# Make Noise 6 same length as respective speech
        X = []
        for sample in speech:
            len_speech = sample.shape[1]
            sample_noise = torch.concat([noise[6],noise[6],noise[6],noise[6]],dim=1)# Repeat noise to ensure noise is longer than speech
            sample_noise = torch.narrow(sample_noise,1,0,len_speech)# Shorten noise to same length as speech
            x = sample_noise
            X.append(x)
        return X

    # STFT
    N_FFT = 1024
    N_HOP = 256

    stft = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT,
        hop_length=N_HOP,
        power=None,
    )

    speech = load_speech()
    noise = load_noise()
    X = add_noise_to_speech(speech, noise)
    X_e = X
    X2 = add_noise_to_speech2(speech,noise)
    noise = get_one_noise(speech,noise)
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
    for n in noise:
        try:
            n_new = stft(n)
            stfts_noise.append(n_new.reshape(513,-1))
        except Exception: #sometimes noises are very short, e.g. noise[697]
            continue

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
    
    X = torch.stack(stfts_mix)
    Y = torch.stack(trainY)
    return X,Y,speech,X_e