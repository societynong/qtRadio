
import funs
import numpy as np
import matplotlib.pyplot as plt
import time

def getParameter():
    global FS,DUR,t,SNR,WIN,STRD,L,N_USED,f0,NFFT

    FS = 10000
    DUR = 60 * 20

    SNR = -40
    WIN = 12000000
    STRD = 12000000
    nOverlap = np.ceil(WIN * (WIN - STRD) / WIN)
    L = int(np.floor((FS*DUR - nOverlap) / (WIN - nOverlap)))
    N_USED = int((WIN - nOverlap) * (L - 1) + WIN)
    f0 = 100
    NFFT = 12000000
    t = np.arange(N_USED) / FS

def getSig():


    x0 = np.sin(2*np.pi*f0*t)
    ns = np.random.randn(N_USED)
    ns = funs.butter_lowpass_filter(ns,10,300,FS)
    ns = ns - np.mean(ns)

    sigPower = 1 / N_USED * np.sum(x0 * x0)

    nsiPower = sigPower / (10**(SNR / 10))

    ns = np.sqrt(nsiPower) / np.std(ns) * ns

    return x0 , ns



getParameter()


def testNormal():
    start = time.time()
    x0, ns0 = getSig()
    sig = x0 + ns0
    # F = funs.accumlateInF(sig, FS, NFFT, WIN, STRD, L).__abs__()
    F = np.abs(np.fft.fft(sig,NFFT))
    end = time.time()
    # print(end - start)
    # plt.figure()
    fP = f0 * NFFT // FS
    plt.plot(abs(F[:300 * NFFT // FS]))
    plt.title(F[fP] / np.mean(np.hstack((F[0:fP - 1], F[fP + 1:300 * NFFT // FS]))))
    plt.show()
    # f = np.arange(funs.NFFT) * funs.FS / funs.NFFT
    # F = abs(funs.accumlateInF(sig,funs.FS,funs.NFFT,funs.WIN,funs.STRD,funs.L))
    # # F = abs(fft(sig,NFFT))
    # plt.figure()
    # plt.plot(f[:300 * funs.NFFT // funs.FS] ,F[:300 * funs.NFFT // funs.FS])
    # plt.show()

def testNoise():
    x0, ns0 = getSig()
    w = 1000
    s = 1000

    sig = x0

    fSig = np.abs(np.fft.fft(sig,FS))


    print(np.max(np.abs(fSig)))
    # sigAcum = np.zeros((1,w))
    # for i in range(0,len(sig) - w,s):
    #     sigAcum += sig[i:i+w]
    # sigAcum /= len(range(0,len(sig) - w,s))
    # print(np.mean(sigAcum.__abs__()))

if __name__ == '__main__':
    testNormal()
