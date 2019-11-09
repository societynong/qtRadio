import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D
from sko.ASFA import ASFA
import funs
DUR = 60*1
FS = 10000
N = int(DUR * FS)
t = np.arange(N) / FS
F0 = 100
FP = 300
N_VIEW = 8
D = 8 * 40
A = 0.5
a = 10000
b = a
SNR = -20
W = 10000
S = 10000


def srU(a,b,x):
    return -a*x + b * x**3

def srDuf(a,b,k,h,sig,f):
    y = np.zeros(len(sig))
    x = np.zeros(len(sig))
    for i in range(len(y) - 1):
        K1 = h * y[i]
        L1 = h * (-k * y[i] - f(a,b,x[i]) + sig[i])
        K2 = h * (y[i] + L1 / 2)
        L2 = h * (-k * (y[i] + L1 / 2) - f(a,b,x[i] + K1 / 2) + sig[i])
        K3 = h * (y[i] + L2 / 2)
        L3 = h * (-k * (y[i] + L2 / 2) - f(a,b,x[i] + K2 / 2) + sig[i + 1])
        K4 = h * (y[i] + L3)
        L4 = h * (-k * (y[i] + L3) - f(a,b,x[i] + K3) + sig[i + 1])
        x[i + 1] = x[i] + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        y[i + 1] = y[i] + 1 / 6 * (L1 + 2 * L2 + 2 * L3 + L4)
    return x

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

def srFunMine(a,b,h,sig):
    u = np.zeros(len(sig))
    for i in range(1,len(u)):
        u[i] =u[i - 1] + h * (a * u[i - 1] - b * u[i - 1] ** 3 + sig[i - 1])
    return u
def getSig(t,F0,FP,FS,SNR):
    x0 = np.sin(2*np.pi*F0*t)
    ns = np.random.randn(N)
    ns = funs.butter_filter(ns,10,FP,FS,'lowpass')
    ns = ns - np.mean(ns)

    sigPower = 1 / N * np.sum(x0 * x0)

    nsiPower = sigPower / (10**(SNR / 10))

    ns = np.sqrt(nsiPower) / np.std(ns) * ns

    return x0 , ns

def getAcmSig(sig,W,S):
    sigAcm = 0
    for i in range(0,len(sig) - W,S):
        sigAcm = sigAcm + sig[i:i+ W]
    return sigAcm / len(range(0,len(sig) - W,S))
def showInF(sig,fMax,fS):
    NFFT = len(sig)
    F_SHOW = int(fMax // (fS / NFFT))
    F_ABS = np.abs(np.fft.fft(sig)[:NFFT // 2]) / len(sig)
    f = np.arange(F_SHOW) / NFFT * FS
    f0 = np.where(f == F0)
    plt.plot(f[:F_SHOW],F_ABS[:F_SHOW])
    plt.title("SNR:{}".format(F_ABS[f0] * (len(F_ABS) - 1) / ((np.sum(F_ABS) - F_ABS[f0]))))
    # plt.title("Max F:{}".format(f[np.argmax(F_ABS[:F_SHOW])]))


def showInT(sig,FS):
    t = np.arange(len(sig)) / FS
    plt.plot(t,sig)



def goalFunc(ab,FS,x0,n0):
    a,b = ab
    sr = srFun(a,b,1 / FS,x0 + n0)
    srF = np.abs(np.fft.fft(sr))[:int(len(sr) / 2)]
    f0N = int(F0 / (FS / len(sr)))

    # sigF = np.abs(np.fft.fft(x0 + n0))[:int(len(x0 + n0) / 2)]

    return srF[f0N] / np.mean(np.hstack((srF[:f0N],srF[f0N:])))
    # return 1 / (srF[f0N] / np.mean(np.hstack((srF[:f0N],srF[f0N:]))) / (sigF[f0N] / np.mean(np.hstack((sigF[:f0N],sigF[f0N:])))))
    # return -np.mean(sr*x0)**2

def goalFuncPSO(ab):
    a,b = ab
    x0 = A * np.sin(2 * np.pi * F0 * t)
    n0 = np.sqrt(2 * D) * np.random.standard_normal(len(x0))
    sr = srFun(a,b,1 / FS,x0 + n0)
    # srF = np.abs(np.fft.fft(sr))[:int(len(sr) / 2)]
    # f0N = int(F0 / (FS / len(sr)))
    # return -srF[f0N] / sum(srF)
    return -np.mean(sr*x0)**2

def goalFuncPSOA(a):
    x0 = 0.01 * np.sin(2 * np.pi * F0 * t)
    n0 = np.sqrt(2 * D) * np.random.standard_normal(len(x0))
    sr = srFun(a,1,1 / FS,x0 + n0)
    # srF = np.abs(np.fft.fft(sr))[:int(len(sr) / 2)]
    # f0N = int(F0 / (FS / len(sr)))
    # return -srF[f0N] / sum(srF)
    return -np.mean(sr*x0)**2

def selIMF(imfs,sig):
    maxV = 0
    sig = sig - np.mean(sig)
    for i in range(imfs.shape[0]):
        imfI = imfs[i]
        imfI = imfI - np.mean(imfI)
        goal = np.sum(imfI * sig) / np.sqrt(np.sum(imfI ** 2) * np.sum(sig ** 2))
        if goal > maxV:
            maxV = goal
    return imfI[i]


def snr(x,n):
    return 10 * np.log10(np.sum(x**2) / np.sum(n ** 2))


def lineMap(mi,mx,sig):
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig)) * (mx - mi) + mi


def test():
    x0, n0 = getSig(t, F0, FP, FS, SNR)
    sig = n0 + x0
    
    # sig = funs.butter_filter(sig,4,(60,150),FS,'bandpass')
    sig = getAcmSig(sig,W,S)
    
    # sig = sig / np.std(sig) * 2
    # sig = lineMap(-1,1,sig) * np.sqrt(4 * a ** 3 / b) * 100000000
    sig = lineMap(-1,1,sig) * np.sqrt(4 * a ** 3 / b / 27) * 40
    sig = sig - np.mean(sig)
    # sig = sig * np.sqrt(4 * a ** 3 / b) #np.sqrt(a ** 3 / b) / np.std(sig) * sig *3
    # sr = srDuf(a,b,0.01,1/FS,sig,srU)
    sr = srFun(a,b,1/FS,sig)
    # sr = getAcmSig(sr,W,S)
    # showInT(sig, FS)
    # showInF(sig, 0.2, FS)

    plt.ion()
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(snr(x0,n0))
    showInT(sig, FS)
    plt.subplot(2, 2, 3)
    showInF(sig, 400, FS)

    plt.subplot(2, 2, 2)
    plt.title(snr(x0, n0))
    showInT(sr, FS)
    plt.subplot(2, 2, 4)
    showInF(sr, 400, FS)

    plt.ioff()
    plt.show()

    # showInT(sr,FS)
    # showInF(sr,0.03,FS)
    # print(optimize.minimize(goalFunc,[0.1,4],(FS,x0,n0),bounds=((0.00001,None),(0.00001,None))))
# 0.00438242,0.00092982

def eggholder(x):
     return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
             -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

def test3D():
    x = np.arange(1, 513)
    y = np.arange(1, 513)
    xgrid, ygrid = np.meshgrid(x, y)
    xy = np.stack([xgrid, ygrid])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, -45)
    ax.plot_surface(xgrid, ygrid, goalFuncPSO(xy), cmap='terrain')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('eggholder(x, y)')
    plt.show()


def snrInF(sig,F0,FS):
    
    NFFT = len(sig)
    sigF = np.abs(np.fft.fft(sig)[:NFFT // 2])
    f = np.arange(NFFT // 2) / NFFT * FS
    f0 = np.where(f == F0)
    return sigF[f0] / (np.sum(sigF) - sigF[f0]) * (len(sigF) - 1)

def exploreWithA():
    x0, n0 = getSig(t, F0, FP, FS, SNR)
    sig = n0 + x0
    sig = getAcmSig(sig,W,S)
    
    gStart = 1
    gStop = 41
    gStep = .5
    gS = np.arange(gStart,gStop,gStep)
    
    aStart = 1
    aStop = 701
    aStep = 10
    aS = np.arange(aStart,aStop,aStep)
    rec = np.zeros((len(aS),len(gS)))
    for ai in range(len(aS)):
        for gi in range(len(gS)):
            aa = aS[ai]
            gg = gS[gi]
            sig1 = lineMap(-1,1,sig) * np.sqrt(4 * aa ** 3 / aa) * gg
            # sig1 = np.sqrt(aa ** 3 / aa)  * sig / np.std(sig) * gg
            sr = srFun(aa,aa,1 / FS,sig1)
            rec[ai][gi] = (float(snrInF(sr,F0,FS)))
            # rec.append(float(snrInF(sr,F0,FS)))
    fig = plt.figure()
    snr0 = float(snrInF(sig,F0,FS))
    giMax,aiMax = np.unravel_index(np.argmax(rec),rec.shape)
    X,Y = np.meshgrid(gS,aS)
    ax = Axes3D(fig)
    ax.plot_surface(X,Y,rec / snr0)
    plt.title("gMax:{},aMax:{},maxGain:{}".format(gS[giMax],aS[aiMax],rec[giMax][aiMax] / snr0))
    sig1 = np.sqrt(aS[aiMax] ** 3 / aS[aiMax])  * sig / np.std(sig) * gS[giMax]
    sr = srFun(aS[aiMax],aS[aiMax],1 / FS,sig1)
    plt.savefig("Opt_SR\SearchMap.png")
    plt.figure()
    plt.subplot(2,1,1)
    showInF(sig,400,FS)
    plt.subplot(2,1,2)
    showInF(sr,400,FS)
    plt.savefig("Opt_SR\Compare.png")
    # maxL = np.argmax(rec)
    # plt.plot(aS,rec)
    # plt.plot(aS,np.ones(len(rec)) * float(snrInF(sig,F0,FS)))
    # plt.title("Max Loc:{},Max Gain:{}".format(aS[maxL],rec[maxL] / float(snrInF(sig,F0,FS))))
    plt.show()
        
if __name__ == "__main__":
    test()