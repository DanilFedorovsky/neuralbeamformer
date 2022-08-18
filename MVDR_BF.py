import numpy as np
from sklearn.decomposition import PCA
import torch

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