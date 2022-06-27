import torch
import torchaudio
import numpy as np
from nara_wpe.wpe import wpe
import os

# Adjust paths to CHiME-5 data directory & output path
datapath = "/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Data/"
savepath = "/Users/danilfedorovsky/Documents/10 Collection/00 Studium/00 Letztes Semester/Masterarbeit/Code/Git Repo/data/1WPE/"
directory = os.fsencode(datapath)

delay = 5
iterations = 5
taps = 10
alpha=0.9999

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        data, sample_rate = torchaudio.load(datapath+filename)
        data = data.detach().numpy()
        data_wpe = wpe(data,taps=taps,delay=delay,iterations=iterations,statistics_mode='full')
        data_wpe = torch.from_numpy(data_wpe)
        torchaudio.save(savepath + "/WPE_"+filename, data_wpe, sample_rate)