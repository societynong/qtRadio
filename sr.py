import numpy as np
import matplotlib.pyplot as plt
DUR = 60 * 1
FS = 10000
N = DUR * FS
t = np.arange(N) / FS
F0 = 100
FP = 300
N_VIEW = 5
SNR = -35

def nView2N(nv,f0,fs):

    return nv * fs // f0

def showInF(sig,fMax,fS):
    NFFT = len(sig)
    F_SHOW = int(fMax // (fS / NFFT))
    F_ABS = np.abs(np.fft.fft(sig))
    f = np.arange(F_SHOW) / NFFT * FS
    plt.figure()
    plt.plot(f[:F_SHOW],F_ABS[:F_SHOW])
    plt.show()

def showInT(sig,FS):
    t = np.arange(len(sig)) / FS
    plt.figure()
    plt.plot(t,sig)
    plt.show()


def test():
    x0 = np.sin(2 * np.pi * F0 * t)
    n0Raw = np.random.standard_normal(N)
    n0 = n0Raw / np.std(n0Raw) * (10 ** (-SNR / 20))

    N_SHOW = nView2N(N_VIEW,F0,FS)
    raw = x0 + n0
    # sr = -1/2*t0**2 + 1 / 4 * t0 ** 4 + t0 * (raw)
    # showInT(raw[:N_SHOW],FS)
    # showInF(sr, FP, FS)
    showInF(raw, FP, FS)


if __name__ == "__main__":
    test()

