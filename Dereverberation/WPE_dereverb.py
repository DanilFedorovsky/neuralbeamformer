import torch
import torchaudio
import numpy as np
from nara_wpe.wpe import wpe
import os
from tqdm import tqdm

# Adjust paths to CHiME-5 data directory & output path
#datapath = "/project/data_asr/CHiME5/data/CHiME5/audio/dev/"
#savepath = "/project/data_asr/CHiME5/data/processed_data/0_WPE"
datapath = "/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/CHiME5/audio/dev/"
savepath = "/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/WPE/"

delay = 5
iterations = 5
taps = 10
alpha=0.9999

directory = os.fsencode(datapath)
for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        data, sample_rate = torchaudio.load(datapath+filename)
        data = data.detach().numpy()
        data_wpe = wpe(data,taps=taps,delay=delay,iterations=iterations,statistics_mode='full')
        data_wpe = torch.from_numpy(data_wpe)
        torchaudio.save(savepath + "/WPE_"+filename, data_wpe, sample_rate)