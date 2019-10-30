import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D
import funs
DUR = 1600
FS = 5
N = int(DUR * FS)
t = np.arange(N) / FS
F0 = 0.005
FP = 300
N_VIEW = 8
D = 0.02
a = 0.1
b = 4.4
def srFun(a,b,h,sig):

    u = np.zeros(len(sig))
    u[0] = -1
    for i in range(len(u) - 1):
        k1 = h * (a * u[i] - b * u[i] ** 3 + sig[i]);
        k2 = h * (a * (u[i] + k1 / 2) - b * (u[i] + k1 / 2) ** 3 + sig[i]);
        k3 = h * (a * (u[i] + k2 / 2) - b * (u[i] + k2 / 2) ** 3 + sig[i + 1]);
        k4 = h * (a * (u[i] + k3) - b * (u[i] + k3) ** 3 + sig[i + 1]);
        u[i + 1] = u[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    # u = u - np.mean(u)
    return u

def showInF(sig,fMax,fS):
    NFFT = len(sig)
    F_SHOW = int(fMax // (fS / NFFT))
    F_ABS = np.abs(np.fft.fft(sig))
    f = np.arange(F_SHOW) / NFFT * FS
    plt.figure()
    plt.plot(f[:F_SHOW],F_ABS[:F_SHOW])
    plt.title("Max F:{}".format(f[np.argmax(F_ABS[:F_SHOW])]))
    plt.show()

def showInT(sig,FS):
    t = np.arange(len(sig)) / FS
    plt.figure()
    plt.plot(t,sig)
    plt.show()


def goalFunc(ab,FS,x0,n0):
    a,b = ab
    sr = srFun(a,b,1 / FS,x0 + n0)
    srF = np.abs(np.fft.fft(sr))[:int(len(sr) / 2)]
    f0N = int(F0 / (FS / len(sr)))

    sigF = np.abs(np.fft.fft(x0 + n0))[:int(len(x0 + n0) / 2)]

    return 1 / (srF[f0N] / np.mean(np.hstack((srF[:f0N],srF[f0N:]))) / (sigF[f0N] / np.mean(np.hstack((sigF[:f0N],sigF[f0N:])))))
    # return -np.mean(sr*x0)**2

def goalFuncPSO(ab):
    a,b = ab
    x0 = 0.01 * np.sin(2 * np.pi * F0 * t)
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
    retI = 0
    maxV = 0
    sig = sig - np.mean(sig)
    for i in range(imfs.shape[0]):
        imfI = imfs[i]
        imfI = imfI - np.mean(imfI)
        goal = np.sum(imfI * sig) / np.sqrt(np.sum(imfI ** 2) * np.sum(sig ** 2))
        if goal > maxV:
            maxV = goal
            retI = i
    return imfI[i]


def test():
    x0 = 0.01 * np.sin(2*np.pi*F0*t)
    n0 = np.sqrt(2 * D) * np.random.standard_normal(len(x0))
    sig = n0 + x0
    res = optimize.minimize(goalFuncPSO,(a,b),bounds=((.001,10),(.001,10)))
    print(res)
    # print(goalFunc((a,b),FS,x0,n0))
    # sr = srFun(a,b,1/FS,sig)
    # showInT(sig, FS)
    # showInF(sig, 0.2, FS)
    # showInT(sr, FS)
    # showInF(sr, 0.2, FS)
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
if __name__ == "__main__":
    test()