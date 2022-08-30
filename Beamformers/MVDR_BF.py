import numpy as np
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

def psdPlt(stfts_mix,speech_pred):
    local_csd = plt.csd(stfts_mix[0][0][0].detach(),speech_pred[0].detach())
    psd_result = torch.add(torch.Tensor(local_csd[0]).reshape(-1,1),torch.Tensor(local_csd[1]).reshape(-1,1))
    for t in range(1,len(stfts_mix[0][0])):
        local_csd = plt.csd(stfts_mix[0][0][t].detach(),speech_pred[t].detach())
        psd_add = torch.add(torch.Tensor(local_csd[0]).reshape(-1,1),torch.Tensor(local_csd[1]).reshape(-1,1))
        psd_result = np.append(psd_result,psd_add.reshape(-1,1),axis=1)
        #psd_result = torch.cat((psd_result,torch.Tensor(local_csd[1]).reshape(-1,1)),dim=1)
    plt.plot(psd_result)
    plt.show()
    return torch.Tensor(psd_result)

def transformToPSD(stfts_mix,speech_pred):
    psd_output = []
    stfts_mix = stfts_mix.reshape(-1,513)
    speech_pred = speech_pred.reshape(-1,513)
    psd_result = torch.zeros((513,1))
    for t in range(0,speech_pred.shape[0]):
        signal_loop = stfts_mix[t].reshape(-1,1)
        speech_loop = speech_pred[t].reshape(-1,1)
        psd_result = torch.add(psd_result, speech_loop*signal_loop*signal_loop.H)
    return psd_result

import numpy as np
from sklearn.decomposition import PCA
import torch

def estimate_d(psd_speech:torch.Tensor):
    """use the principal component of the estimated power spectral
    density matrix of speech: d = P {ΦXX}."""
    phi_speech = psd_speech.detach().numpy().reshape(513,-1)
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(phi_speech)
    return principal_component

def mvdr_beamformer(d: np.array, psd_noise:torch.Tensor, spectrum: torch.Tensor):
    """
    inputs:
    - steering vector d (Nx1)
    - sigma noise (NxN)
    - spectrum (in this bin, TXN)
    """
    psd_noise = psd_noise.reshape(513,-1)
    d = np.matrix(d) # 1xN => Nx1 (513x1)
    spectrum = np.matrix(spectrum[0].detach().numpy()).T # TxN => NxT
    phiNN_inv = np.matrix(psd_noise.detach().numpy()).I # NxN
    w = phiNN_inv * d / (d.H * phiNN_inv * d)
    result_spectrum = w.H * spectrum.T
    return result_spectrum

def estimate_d(speech:torch.Tensor):
    """use the principal component of the estimated power spectral
    density matrix of speech: d = P {ΦXX}."""
    phi_speech = speech.detach().numpy().reshape(513,-1)
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(phi_speech)
    return principal_component

def mvdr_beamformer(d, noise, spectrum):
    """
    inputs:
    - steering vector d (Nx1)
    - sigma noise (NxN)
    - spectrum (in this bin, TXN)
    """
    d = np.matrix(d).T # 1xN => Nx1
    spectrum = np.matrix(spectrum).T # TxN => NxT
    phiNN_inv = np.matrix(noise).I

    w = phiNN_inv * d / (d.H * phiNN_inv * d)
    result_spectrum = w.H * spectrum
    return result_spectrum