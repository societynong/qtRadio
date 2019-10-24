import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter
from numpy.fft import fft

def choose_windows(name,N):
    if name == 'Rect':
        window = np.ones(N)
    elif name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    return window




def butter_lowpass_filter(data, order, cutoff, FS):
    b,a = butter(order,2 * cutoff / FS,'low')
    return lfilter(b,a,data)







def accumlateInF(sig,FS,NFFT,WIN,STRD):
    f = np.arange(NFFT) * FS / NFFT
    fAccum = 0
    for n in range(0,len(sig),STRD):
        nstart = n * STRD
        dw = 2 * np.pi * f * nstart / FS
        dpp = np.exp(-1j * dw)

        subSig = sig[nstart:nstart + WIN]
        subF = fft(subSig,NFFT) / NFFT
        fAccum += subF * dpp

    return fAccum / len(range(0,len(sig),STRD))