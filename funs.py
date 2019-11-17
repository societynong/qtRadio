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

def getAcmSig(sig,W,S):
    sigAcm = 0
    for i in range(0,len(sig) - W,S):
        sigAcm = sigAcm + sig[i:i+ W]
    return sigAcm / len(range(0,len(sig) - W,S))

def getMaxF(sig,FS):
    NFFT = len(sig)
    f = np.arange(NFFT) / NFFT * FS
    sigF = np.abs(np.fft.fft(sig))[:int(NFFT // 2)]
    return f[np.argmax(sigF)]


def getSig(t,F0,FP,FS,SNR):
    x0 = np.sin(2*np.pi*F0*t)
    ns = np.random.randn(len(t))
    ns = butter_filter(ns,10,FP,FS,'lowpass')
    ns = ns - np.mean(ns)
    sigPower = 1 / len(t) * np.sum(x0 * x0)
    nsiPower = sigPower / (10**(SNR / 10))
    ns = np.sqrt(nsiPower) / np.std(ns) * ns
    return x0 , ns

def lineMap(mi,mx,sig): 
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig)) * (mx - mi) + mi

def showInF(sig,fMax,fS):
    NFFT = len(sig)
    F_SHOW = int(fMax // (fS / NFFT))
    F_ABS = np.abs(np.fft.fft(sig)[:NFFT // 2]) / len(sig)
    f = np.arange(F_SHOW) / NFFT * fS
    plt.plot(f[:F_SHOW],F_ABS[:F_SHOW])
    plt.xlabel("Hz")
    # plt.title("Max F:{}".format(f[np.argmax(F_ABS[:F_SHOW])]))

def showInFDetail(sig,fMax,fS,fGoal):
    NFFT = len(sig)
    F_SHOW = int(fMax // (fS / NFFT))
    F_ABS = np.abs(np.fft.fft(sig)[:NFFT // 2]) / len(sig)
    f = np.arange(F_SHOW) / NFFT * fS
    f0 = np.where(np.abs(f - fGoal) < fS / NFFT / 2 )
    plt.plot(f[:F_SHOW],F_ABS[:F_SHOW])
    plt.xlabel("Hz")
    plt.title("Signal weight:{} Highest frequency location:{}Hz".format(F_ABS[f0] * (len(F_ABS) - 1) / ((np.sum(F_ABS) - F_ABS[f0])),f[np.argmax(F_ABS[:F_SHOW])]))
    # plt.title("Max F:{}".format(f[np.argmax(F_ABS[:F_SHOW])]))


def snr(x,n):
    return 10 * np.log10(np.sum(x**2) / np.sum(n ** 2))

def srFun(a,b,h,sig):
    u = np.zeros(len(sig))
    for i in range(len(u) - 1):
        k1 = h * (a * u[i] - b * u[i] ** 3 + sig[i])
        k2 = h * (a * (u[i] + k1 / 2) - b * (u[i] + k1 / 2) ** 3 + sig[i])
        k3 = h * (a * (u[i] + k2 / 2) - b * (u[i] + k2 / 2) ** 3 + sig[i + 1])
        k4 = h * (a * (u[i] + k3) - b * (u[i] + k3) ** 3 + sig[i + 1])
        u[i + 1] = u[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    # u = u - np.mean(u)
    return u

def getInF(sig,fS):
    fMaxN = int(len(sig) // 2)
    sigF = np.abs(np.fft.fft(sig))[:fMaxN] / len(sig)
    f = np.arange(fMaxN) / fMaxN * fS
    return sigF,f

