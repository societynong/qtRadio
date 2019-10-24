
import funs
import numpy as np
import matplotlib.pyplot as plt
import time

FS = 10000
DUR = 60 * 1

SNR = -20
WIN = 10000
STRD = 10000
N_USED = DUR * FS
f0 = 100
NFFT = 12000000


def getSig():
    t = np.arange(N_USED) / FS
    x0 = np.sin(2*np.pi*f0*t)
    ns = np.random.randn(N_USED)
    ns = funs.butter_lowpass_filter(ns,10,300,FS)
    ns = ns - np.mean(ns)

    sigPower = 1 / N_USED * np.sum(x0 * x0)

    nsiPower = sigPower / (10**(SNR / 10))

    ns = np.sqrt(nsiPower) / np.std(ns) * ns

    return x0 , ns






def testNormal():
    start = time.time()
    x0, ns0 = getSig()
    sig = x0 + ns0
    # F = funs.accumlateInF(sig, FS, NFFT, WIN, STRD, L).__abs__()
    F = np.abs(np.fft.fft(sig,NFFT))  / len(sig)
    end = time.time()
    # print(end - start)
    # plt.figure()
    fP = f0 * NFFT // FS
    f = np.arange(300 * NFFT // FS) / (300 * NFFT // FS) * 300
    plt.plot(f,abs(F[:300 * NFFT // FS]))
    plt.xlabel("Hz")
    plt.title("Mean of Noise:{:.3f},Mean of Signal{:.3f},Time:{}min(s)".format(np.mean(np.hstack((F[0:fP - 1], F[fP + 1:300 * NFFT // FS]))),F[fP],DUR / 60))
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
def testTandSNR(TRef,SNRRef,fd):
    global DUR,SNR,N_USED
    DUR = 60 * TRef
    SNR = SNRRef
    N_USED = DUR * FS
    x0, ns0 = getSig()
    sig = x0 + ns0
    # F = funs.accumlateInF(sig, FS, NFFT, WIN, STRD, L).__abs__()
    F = np.abs(np.fft.fft(sig,NFFT))  / len(sig)
    fP = f0 * NFFT // FS
    f = np.arange(300 * NFFT // FS) / (300 * NFFT // FS) * 300
    plt.plot(f,abs(F[:300 * NFFT // FS]))
    plt.xlabel("Hz")
    plt.title("Mean of Noise:{:.3f},Mean of Signal{:.3f},Time:{}min(s)".format(np.mean(np.hstack((F[0:fP - 1], F[fP + 1:300 * NFFT // FS]))),F[fP],TRef))
    # plt.show()
    plt.savefig("TandSNR/test_{}_{}.png".format(TRef,SNRRef))
    plt.close()

    fd.write("{}\t{}\t{}\t{}\n".format(TRef,SNRRef,np.mean(np.hstack((F[0:fP - 1], F[fP + 1:300 * NFFT // FS]))),F[fP]))


if __name__ == '__main__':
    Ts = [1,2,4,8,16,32,64]
    SNRs = [-10,-20,-30,-40,-50]
    with open("TandSNR/ExploreTandSNR.txt","a+") as fd:
        cnt = 0
        for snr in SNRs:
            testTandSNR(8,snr,fd)
            cnt += 1
            print("{} ok!".format(cnt / len(SNRs)))


