import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from pyhht import EMD
from pyhht.visualization import plot_imfs
import funs
DUR = 60 * 1
FS = 10000
N = int(DUR * FS)
t = np.arange(N) / FS
F0 = 100
FP = 300
N_VIEW = 8
SNR = -30

def getSig(t,F0,FP,FS,SNR):
    x0 = np.sin(2*np.pi*F0*t)
    ns = np.random.randn(N)
    ns = funs.butter_lowpass_filter(ns,10,FP,FS)
    ns = ns - np.mean(ns)

    sigPower = 1 / N * np.sum(x0 * x0)

    nsiPower = sigPower / (10**(SNR / 10))

    ns = np.sqrt(nsiPower) / np.std(ns) * ns

    return x0 , ns

def srFun(a,b,h,sig):
    u = np.zeros((len(sig)))
    for i in range(len(u) - 1):
        k1 = h * (a * u[i] - b * u[i] ** 3 + sig[i]);
        k2 = h * (a * (u[i] + k1 / 2) - b * (u[i] + k1 / 2) ** 3 + sig[i]);
        k3 = h * (a * (u[i] + k2 / 2) - b * (u[i] + k2 / 2) ** 3 + sig[i + 1]);
        k4 = h * (a * (u[i] + k3) - b * (u[i] + k3) ** 3 + sig[i + 1]);
        u[i + 1] = u[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    u = u - np.mean(u)
    return u
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

def goalFunc(ab,FS,x0,n0):
    a,b = ab
    sr = srFun(a,b,1 / FS,x0 + n0)
    return -np.mean(sr*x0)**2

def test():
    x0,n0 = getSig(t, F0, FP, FS, SNR)
    # N_SHOW = nView2N(N_VIEW,F0,FS)
    sr = srFun(0.00438242,0.00092982,1 / FS,x0 + n0)
    showInF(x0 + n0,300,FS)
    showInF(sr,300,FS)
    # print(optimize.minimize(goalFunc,[1,1],args=(FS,x0,n0)))


def fy(t):
    return t**2

def testOpt():
    print(optimize.fsolve(fy,3))
if __name__ == "__main__":
    test()

