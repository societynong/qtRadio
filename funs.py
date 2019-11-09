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




def butter_filter(data, order, cutoff, FS,type):
    if type == 'lowpass' or type == 'highpass':
        b,a = butter(order,2 * cutoff / FS,type)
    elif type == 'bandpass':
        b,a = butter(order,(2 * cutoff[0] / FS, 2 * cutoff[1] / FS),type)
    else :
        return None
    return lfilter(b,a,data)







def accumlateInF(sig,FS,NFFT,WIN,STRD):
    f = np.arange(NFFT) * FS / NFFT
    fAccum = 0
    for n in range(0,len(sig) - WIN,STRD):
        nstart = n
        dw = 2 * np.pi * f * nstart / FS

        dpp = np.exp(-1j * dw)

        subSig = sig[nstart:nstart + WIN]
        subF = fft(subSig,NFFT)
        fAccum += subF * dpp
        # plt.plot(np.real(subSig))
        # plt.show()


    return fAccum / len(range(0,len(sig) - WIN,STRD))