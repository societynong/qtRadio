import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm,tree
from sklearn.neural_network import MLPClassifier
import funs
import pickle as pkl
from sklearn.model_selection import train_test_split
import os
import sys

DUR = 60*20
FS = 10000
N = int(DUR * FS)
t = np.arange(N) / FS
F0 = 100
FP = 300
FA = 98
N_VIEW = 8
D = 8 * 40
A = 0.6
a = 400
b = a
SNR = -50
win = 10000
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
    ns = np.random.randn(len(t))
    ns = funs.butter_filter(ns,10,FP,FS,'lowpass')
    ns = ns - np.mean(ns)

    sigPower = 1 / len(t) * np.sum(x0 * x0)

    nsiPower = sigPower / (10**(SNR / 10))
    ns = np.sqrt(nsiPower) / np.std(ns) * ns

    return x0 , ns

def getNoiseFix(t,F0,FP,FS,SNR):
    x0 = np.sin(2*np.pi*F0*t)
    ns = np.random.randn(len(t))
    ns = funs.butter_filter(ns,10,FP,FS,'highpass')
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
    plt.plot(f[:F_SHOW],F_ABS[:F_SHOW])
    plt.xlabel("Hz")
    # plt.title("Max F:{}".format(f[np.argmax(F_ABS[:F_SHOW])]))

def showInFDetail(sig,fMax,fS,fGoal):
    NFFT = len(sig)
    F_SHOW = int(fMax // (fS / NFFT))
    F_ABS = np.abs(np.fft.fft(sig)[:NFFT // 2]) / len(sig)
    f = np.arange(F_SHOW) / NFFT * FS
    f0 = np.where(np.abs(f - fGoal) < fS / NFFT / 2 )
    plt.plot(f[:F_SHOW],F_ABS[:F_SHOW])
    plt.xlabel("Hz")
    plt.title("Signal weight:{} Highest frequency location:{}Hz".format(F_ABS[f0] * (len(F_ABS) - 1) / ((np.sum(F_ABS) - F_ABS[f0])),f[np.argmax(F_ABS[:F_SHOW])]))
    # plt.title("Max F:{}".format(f[np.argmax(F_ABS[:F_SHOW])]))

def showInT(sig,FS):
    t = np.arange(len(sig)) / FS
    plt.plot(t,sig)
    plt.xlabel("s")


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

def getAcumOverlap(sig,W,S):
    sigAcm = 0
    for i in range(0,len(sig) - W,S):
        # startLoc = 1 + (i - 1) * S
        tmp = np.zeros(W)
        for j in range(i,i+W):
            tmp[j % W] = sig[j]
        sigAcm = sigAcm + tmp
    return sigAcm

def getMaxF(sig,FS):
    NFFT = len(sig)
    f = np.arange(NFFT) / NFFT * FS
    sigF = np.abs(np.fft.fft(sig))[:int(NFFT // 2)]
    return f[np.argmax(sigF)]

def test():
    TT = 20
    accRW = 0
    accTZ = 0
    accN = 0
    for tt in range(TT):
        x0, n0 = getSig(t, F0, FP, FS, SNR)
        sig0 = n0 + x0
        
        # sig = funs.butter_filter(sig,4,(60,150),FS,'bandpass')
        
        sigTZ = getAcmSig(sig0 * np.cos(2 * np.pi * FA * t),win,S)
        nTZ = getAcmSig(n0 * np.cos(2 * np.pi * FA * t),win,S)
        sigRW = getAcmSig(sig0,win,S)
        # sig = sig / np.std(sig) * 2
        # sig = lineMap(-1,1,sig) * np.sqrt(4 * a ** 3 / b) * 100000000
        sigTZ = lineMap(-1,1,sigTZ) * np.sqrt(4 * a ** 3 / b / 27) * 20
        nTZ = lineMap(-1,1,nTZ) * np.sqrt(4 * a ** 3 / b / 27) * 20
        sigTZ = sigTZ - np.mean(sigTZ)
        nTZ = nTZ - np.mean(nTZ)
        # sig = sig * np.sqrt(4 * a ** 3 / b) #np.sqrt(a ** 3 / b) / np.std(sig) * sig *3
        # sr = srDuf(a,b,0.01,1/FS,sig,srU)
        srTZ = srFun(a,b,1/FS,sigTZ)
        srN = srFun(a,b,1/FS,nTZ)
        # sr = getAcmSig(sr,W,S)
        # showInT(sig, FS)
        # showInF(sig, 0.2, FS)
        if np.abs(getMaxF(sigRW,FS) - F0) <= FS / len(sigRW) / 2:
            accRW += 1
        if np.abs(getMaxF(srTZ,FS) - (F0 - FA)) <= FS / len(srTZ) / 2:
            accTZ += 1
        if np.abs(getMaxF(srN,FS) - (F0 - FA)) <= FS / len(srN) / 2:
            accN += 1
    print("RW:{}".format(accRW / TT))
    print("TZ:{}".format(accTZ / TT))
    print("N:{}".format(accN / TT))
####################################################
    # plt.ion()
    # plt.figure('Test',figsize=(16,12))
    # plt.subplot(2, 2, 1)
    # plt.title(snr(x0,n0))
    # showInT(sigRW, FS)
    # plt.subplot(2, 2, 3)
    # showInFDetail(sigRW, 400, FS,F0)

    # plt.subplot(2, 2, 2)
    # plt.title(snr(x0, n0))
    # showInT(srTZ, FS)
    # plt.subplot(2, 2, 4)
    # showInFDetail(srTZ, 30, FS,F0 - FA)

    # plt.ioff()
    # plt.show()
####################################################
    # showInT(sr,FS)
    # showInF(sr,0.03,FS)
    # print(optimize.minimize(goalFunc,[0.1,4],(FS,x0,n0),bounds=((0.00001,None),(0.00001,None))))
    # print(optimize.minimize(goalFunc,[0.1,4],(FS,x0,n0),bounds=((0.00001,None),(0.00001,None))))
    # print(optimize.minimize(goalFunc,[0.1,4],(FS,x0,n0),bounds=((0.00001,None),(0.00001,None))))
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
    f0 = np.where(f == F0 - FA)
    return sigF[f0] / (np.sum(sigF) - sigF[f0]) * (len(sigF) - 1)


def snrTest():
    D = 1
    n0 = np.sqrt(2 * D) * np.random.rand(len(t))
    x0 = 0.1 * np.sin(2 * np.pi * F0 * t)
    print(snr(x0,n0))
    print(snr(x0,n0))
    print(snr(x0,n0))
    print(snr(x0,n0))

def exploreWithA():
    x0, n0 = getSig(t, F0, FP, FS, SNR)
    sig = n0 + x0
    sig = getAcmSig(sig,win,S)
    
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


def jiangpin(sig,f0,f1):
    n = int(f0 / f1)
    m = int(len(sig) / n)
    ret = sig[:m*n]
    ret = np.reshape(sig,(n,m))
    ret = np.transpose(ret,(1,0))
    ret = np.reshape(ret,m*n)
    return ret


def _testPoint(sig,f0):
    # sig = lineMap(-1,1,sig)
    sigAcm = sig #getAcmSig(sig,W,S)
    sigAcm = sigAcm - np.mean(sigAcm)
    sigAcm = sigAcm / np.std(sigAcm)
    NFFT = len(sigAcm)
    sigAcmF = np.abs(np.fft.fft(sigAcm))[:int(NFFT // 2)]
    f = np.arange(int(NFFT // 2))  / NFFT * FS
    # plt.plot(f,sigAcmF)
    # plt.xlim([0,300])
    # plt.show()
    return sigAcmF[np.where(np.abs(f - f0) <= FS / NFFT / 2)]

def getAcmFeature(sig,W,S,fs,f0):
    feature = []
    sigAcm = np.zeros(W)
    df = fs / W
    f = np.arange(W) / W * fs
    argF0 = np.where(np.abs(f - f0) <= df / 2)[0][0]
    for i in range(0,len(sig) - 1,S):
        sigW = sig[i:i+W]
        pre = np.abs(np.fft.fft(sigAcm))[argF0]
        sigAcm = sigAcm + sigW
        nw = np.abs(np.fft.fft(sigAcm))[argF0]
        feature.append(nw - pre)
    return feature

def getAcmHalFeature(sig,fs,f0):
    sig = sig[:int(len(sig) / f0) * f0]
    W = int(fs / f0 // 2)
    feature = 0
    flag = 1
    for i in range(0 , len(sig) - W, W):
        feature += flag * sig[i:i+W] 
        flag *= -1
    return (feature / len(range(0 , len(sig) - W, W))).tolist()


def getFinalFeature(sig,fS,f0):
    sigF = np.fft.fft(sig)
    df = fS / len(sigF)
    f = np.arange(len(sig)) / len(sig) * FS
    f2N = np.where(np.abs(f - f0) < df / 2)[0][0]
    sigF0 = np.zeros(len(sigF),dtype=complex)
    sigF0[f2N] = sigF[f2N]
    sigF0[-f2N] = sigF[-f2N]
    
    featureFinal = np.real(np.fft.ifft(sigF0))
    featureFinal = getAcmSig(featureFinal,200,200)
    return featureFinal.tolist()

def dataGeneration(start,stop,step,featureName):
    for snr in range(start,stop,step):
        filetoSave = "features\\test\\{}{}.pkl".format(snr,featureName)
        if os.path.exists(filetoSave):
            continue
        nPoints = 200
        fN0Rec = []
        fSigRec = []
        for i in range(nPoints):
            x0, n0 = getSig(t, F0, FP, FS, snr)
            # fN0 = getAcmSig(n0,200,200)
            # sig = (sig - sig[np.arange(len(sig) - 1,-1,-1)]) / 2
            # sig = sig - np.mean(sig)
            # sigSr = srFun(a,b,1/FS,sig)
            
            # fN0 = _testPoint(sig,F0)
            # fX0 = _testPoint(x0,F0)
            # fN0 = getAcmFeature(n0,10000,10000,FS,F0)
            # fN0 = getAcmHalFeature(n0,FS,F0)
            # fN0 = getSTFTFeature(n0,FS,10000)
            # fN0 = getFakeFeature(n0,FS,F0,8)
            fN0 = getFinalFeature(n0,FS,F0)
            x0, n0 = getSig(t, F0, FP, FS, snr)
            # fSig = getAcmSig(x0 + n0,200,200)            
            # sig = (sig - sig[np.arange(len(sig) - 1,-1,-1)]) / 2
            # sig = sig - np.mean(sig)
            # sigSr = srFun(a,b,1/FS,sig)
            # fSig = _testPoint(sig,F0)
            # fSig = getAcmFeature(x0 + n0,10000,10000,FS,F0)
            # fSig = getAcmHalFeature(x0 + n0,FS,F0)
            # fSig = getSTFTFeature(x0 + n0,FS,10000)
            # fSig = getFakeFeature(x0+n0,FS,F0,8)\
            fSig = getFinalFeature(x0 + n0,FS,F0)
            # plt.figure()
            # plt.subplot(2,1,1)
            # showInT(fN0,FS)
            # plt.subplot(2,1,2)
            # showInT(fSig,FS)
            # plt.show()
            fN0Rec.append(fN0)
            fSigRec.append(fSig)
            print("{}% of {}db".format(i / nPoints * 100,snr))
        X = np.array(fN0Rec+fSigRec)
        y = np.array(([0] * len(fN0Rec))+([1] * len(fSigRec)))
        with open(filetoSave,'wb') as f:
            pkl.dump((X,y),f)


from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.model_selection import cross_val_score
# from deepnetwork import FeatureNet
def testSinglePoint(start,stop,step,featureName):
    # nPoints = 40
    # plt.figure()
    # plt.title("SNR:{}".format(SNR))
    # fN0Rec = []
    # fSigRec = []
    # for i in range(nPoints):
    #     x0, n0 = getSig(t, F0, FP, FS, SNR)
    #     fN0 = _testPoint(n0,F0)
    #     fX0 = _testPoint(x0,F0)
    #     fSig = _testPoint(x0 + n0,F0)
    #     # print("x0:{},n0:{},x0 + n0:{}".format(fX0,fN0,fSig))
    #     fN0Rec.append(fN0)
    #     fSigRec.append(fSig)
    #     plt.scatter(fN0,fN0,c='red')
    #     plt.scatter(fSig,fSig,c='green')
    #     print("{}%".format(i / nPoints * 100))

    # # plt.savefig("{}db.png".format(SNR))
    # plt.show()
    # X = np.reshape(np.array(fN0Rec+fSigRec),[-1,1])
    # y = np.array(([0] * len(fN0Rec))+([1] * len(fSigRec)))
    # model = svm.SVC(kernel='linear')

    for snr in range(start,stop,step):
        with open("features\\test\\{}{}.pkl".format(snr,featureName),'rb') as f:
            X,y = pkl.load(f)
        rIdx = np.array(list(range(len(X))))
        np.random.shuffle(rIdx)
        X = X[rIdx]
        y = y[rIdx]
        # fnw = FeatureNet().cuda()
        # StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=3)
        # fnw.fit(X_train.reshape([X_train.shape[0],1,X_train.shape[1],X_train.shape[2]]),y_train)
        # # X = fnw(np.concatenate(X.reshape([X.shape[0],1,X.shape[1],X.shape[2]])))
        # y_pred = fnw(X_test.reshape([X_test.shape[0],1,X_test.shape[1],X_test.shape[2]])).cpu().detach().numpy()
        # print(y_pred)
        # print(y_test)
        # print("Accuracy:{}".format(np.mean(y_pred == y_test)))
      
        
        
        model = svm.SVC(kernel='linear',gamma='auto')#tree.DecisionTreeClassifier()
        # model = MLPClassifier()
        model.fit(X_train,y_train)
        acc = cross_val_score(model,X_test,y_test,cv=5,scoring='accuracy')#model.score(X_test,y_test)
        print("{} db, Acc:{},Mean:{}".format(snr,acc,np.mean(acc)))
def jiangpinDemo():
    x0, n0 = getSig(t, F0, FP, FS, SNR)
    # _,n1 = getNoiseFix(t,F0,FP,FS,SNR)
    
    sig0 = n0
    # sig1 = sig0 * np.cos(2 * np.pi * 99 * t)
    sig1 = jiangpin(sig0,100,1)
    plt.figure()
    plt.subplot(2,2,1)
    showInT(sig0[:10000],FS)
    plt.subplot(2,2,3)
    showInF(sig0,600,FS)
    plt.subplot(2,2,2)
    showInT(sig1[:10000],FS)
    plt.subplot(2,2,4)
    showInF(sig1,600,FS)
    plt.show()


def analysisOverlap():
    N = 1000
    T = 100
    sig = np.arange(N) % T
    plt.figure()
    showInT(sig,1)
    plt.show()

    plt.figure()
    showInT(sig[20:20 + T],1)
    plt.show()

    W = T
    sigF = np.fft.fft(sig[20:20 + W])
    sigFFix = sigF * np.exp(-1j * 2 * np.pi * np.arange(W) / W * 20)
    sigFix = np.real(np.fft.ifft(sigFFix))
    plt.figure()
    showInT(sigFix,1)
    plt.show()


def testInT():
    x0,n0 = getSig(t, F0, FP, FS, -50)
    sig = getAcmSig( n0,200,200)
    sig1 = (sig - sig[np.arange(len(sig) - 1,-1,-1)]) / 2
    plt.figure()
    plt.subplot(2,1,1)
    showInT(sig,FS)
    plt.subplot(2,1,2)
    showInT(sig1,FS)
    plt.show()

from scipy.signal import stft
# import cv2
from scipy import misc

def getSTFTFeature(sig,FS,win):
    f, tt, Zxx = stft(sig, FS, nperseg=win)
    Zxx = abs(Zxx)
    df = FS / win
    f = f[:int(FP / df) ]
    Zxx = Zxx[:int(FP / df) ][:]
    # imgZxx = np.uint16(lineMap(0,2 ** 16 - 1,Zxx))
    imgZxx = lineMap(0,1,Zxx)
    imgZxx = cv2.resize(imgZxx,(96,96))
    return imgZxx

def testSTFT():
    
    x0,n0 = getSig(t,F0,FP,FS,-38)
    sig = n0 + x0
    sig = getAcmSig(sig,100000,100000)
    win = 10000
    f, tt, Zxx = stft(sig, FS, nperseg=win,noverlap=win // 2)
    Zxx = abs(Zxx)
    df = FS / win
    f = f[:int(FP / df) ]
    Zxx = Zxx[:int(FP / df) ][:]
    # imgZxx = np.uint16(lineMap(0,2 ** 16 - 1,Zxx))
    imgZxx = lineMap(0,1,Zxx)
    imgZxx = cv2.resize(imgZxx,(96,96))
    plt.imshow(imgZxx)
    plt.show()


def getSigEnhanced(sig,fS,f0):
    nT = int(fS // f0)
    hfT = int(nT // 2)
    sigHelp = sig
    for i in range(0,len(sig) - nT,nT):
        # plt.figure()
        sigHelp[i : i+hfT] = -sig[i+hfT:i+nT]
        sigHelp[i+hfT:i + nT] = -sig[i : i+hfT]
        # plt.plot(sigHelp[i:i+nT])
        # plt.show()
    return (sig + sigHelp) / 2
    

def getAcmSigEnhanced(sig,win,fS,f0):
    nT = int(fS // f0)
    nTW = int(win // nT)
    sigAcm = 0
    epoch = 2

    for _ in range(epoch):
        idx = np.arange(win)
        idxtmp = np.arange(win)
        order = np.arange(nTW)
        np.random.shuffle(order)
        for i in range(0,len(sig) - win,win):
            
            for idxo in range(len(order)):
                idx[idxo * nT:(idxo + 1) * nT] = idxtmp[order[idxo] * nT:(order[idxo] + 1)*nT]
            sigAcm = sigAcm + (sig[i:i+win])[idx]
    return sigAcm / epoch

            
def generateFakeSig(sig,fS,f0):
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    order = np.arange(nW)
    np.random.shuffle(order)
    fakeSig = np.zeros(len(sig))
    for idxo in range(int(len(order))):
        if np.random.randint(50) % 2 == 0:
            fakeSig[idxo * nT : (idxo + 1) * nT] = sig[order[idxo] * nT : (order[idxo] + 1) * nT]
        else:
            fakeSig[idxo * nT : (idxo + 1) * nT] = -(sig[order[idxo] * nT : (order[idxo] + 1) * nT])[::-1]
    
    # nT = int(fS // f0)
    # nW = int(len(sig) // nT)
    # order = np.arange(nW * 4)
    # np.random.shuffle(order)
    # fakeSig = np.zeros(len(sig))
    # part4 = int(nT // 4)
    # for idxo in range(len(order)):
    #     od = order[idxo]
        
    #     idxo4 = idxo % 4
    #     od4 = od % 4
    #     if idxo4 == od4:
    #         sigPart4 = sig[od * part4 : (od + 1) * part4]
    #     elif (idxo4 == 0 and od4 == 1) or (idxo4 == 2 and od4 == 3) or (idxo4 == 1 and od4 == 0) or (idxo4 == 3 and od4 == 2):
    #         sigPart4 = (sig[od * part4 : (od + 1) * part4])[::-1]
    #     elif (idxo4 == 0 and od4 == 2) or (idxo4 == 1 and od4 == 3) or (idxo4 == 2 and od4 == 0) or (idxo4 == 3 and od4 == 1):
    #         sigPart4 = -sig[od * part4 : (od + 1) * part4]
    #     else:
    #         sigPart4 = -(sig[od * part4 : (od + 1) * part4])[::-1]
    #     fakeSig[idxo * part4 : (idxo + 1) * part4] = sigPart4

    return fakeSig


def getFakeFeature(sig,fS,f0,epoch):
    sigAcm = fakeAcm(sig,fS,f0,epoch)
    sigAcm200 = getAcmSig(sigAcm,200,200)
    sigAcm10000 = getAcmSig(sigAcm,10000,10000)
    _testPoint(sigAcm10000,f0)
    # sigAcm = sigAcm - np.mean(sigAcm)
    # sigAcm = sigAcm / np.std(sigAcm)
    return sigAcm200.tolist() + _testPoint(sigAcm10000,f0)

def fakeAcm(sig,fS,f0,epoch):
    sigAcm = 0
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    orders = np.zeros((epoch,nT,nW),np.int)
    for i in range(epoch):
        for j in range(nT):
            orders[i][j] = np.arange(int(len(sig) / fS * f0))
            np.random.shuffle(orders[i][j])
    
    with open('order\\{}.pkl'.format(epoch),'rb') as f:
        orders = pkl.load(f)
    
    for e in range(epoch):
        # sigAcm = sigAcm + generateFakeSigV2(sig,fS,f0)
        sigAcm = sigAcm + generateFakeSigV4(sig,fS,f0,orders[e])
    with open('order\\{}.pkl'.format(epoch),'wb') as f:
        pkl.dump(orders,f)
    return sigAcm / epoch

def generateFakeSigV4(sig,fS,f0,order):
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    fakeSig = np.zeros(len(sig))
    for nt in range(nT):
        for nw in range(nW):
            fakeSig[nw * nT + nt] = sig[order[nt][nw] * nT + nt]
    return fakeSig

def generateFakeSigV3(sig,fS,f0,order):
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    fakeSig = np.zeros(len(sig))
    idx = np.arange(nW)
    for iT in range(nT):
        fakeSig[idx * nT + iT] = sig[order * nT + iT] 
    return fakeSig
def generateFakeSigV2(sig,fS,f0):
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    fakeSig = np.zeros(len(sig))
    idx = np.arange(nW)
    for iT in range(nT):
        order = np.arange(nW)
        np.random.shuffle(order)
        fakeSig[idx * nT + iT] = sig[order * nT + iT] 
    return fakeSig

def getAcmF(sig,w,s):
    fAcm = 0
    for i in range(0,len(sig) - w,s):
        fAcm += np.abs(np.fft.fft(sig[i:i+w]))
    return fAcm / len(range(0,len(sig) - w,s))


def getAcmOpt(n,epoch):
    fakeN = 0
    for _ in range(epoch):
        idx = np.arange(len(n))
        np.random.shuffle(idx)
        fakeN += n[idx]
    return fakeN / epoch
def testFake():
    # while True:
    epoch = 4 #----9db
    win = 10000
    x0,n0 = getSig(t[:10000 * 60 * 20],F0,FP,FS,-60)
    print(snr(x0,n0))
    # n1 = (fakeAcm(n0,FS,F0,epoch) - fakeAcm(n0[::-1],FS,F0,epoch)) / 2
    n1 = fakeAcm(n0,FS,F0,epoch)
    # n1 = getAcmOpt(n0,epoch)
    n1 = (n1 - n1[::-1]) / 2
    # n1 = (n1 - n1[::-1]) / 2
    sig1 = n1 + x0
    sig1Acm = getAcmSig(sig1,win,win)
    n1Acm = getAcmSig(n1,win,win)
    plt.figure()
    plt.subplot(2,1,1)
    showInFDetail(sig1Acm,FP,FS,F0)
    plt.subplot(2,1,2)
    showInFDetail(n1Acm,FP,FS,F0)
    plt.show()
    # sig1AcmF = getAcmF(sig1,win,win)[:int(FP / (FS / win))]
    # n1AcmF = getAcmF(n1,win,win)[:int(FP / (FS / win))]
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(sig1AcmF)
    # plt.subplot(2,1,2)
    # plt.plot(n1AcmF)

    # x1 = getAcmSig(x0,10000,10000)
    # n1AcmF = np.abs(np.fft.fft(n1Acm))[:int(FP / (FS / len(n1Acm)))]
    # agn = np.angle(np.fft.fft(n1Acm)[int(F0 / (FS / len(n1Acm)))]) / np.pi * 180
    # agx = np.angle(np.fft.fft(x1)[int(F0 / (FS / len(x1)))]) / np.pi * 180
    # apn = n1AcmF[int(F0 / (FS / len(n1Acm)))]
    # apn / np.mean(np.concatenate((n1AcmF[:int(F0 / (FS / len(n1Acm)))],n1AcmF[int(F0 / (FS / len(n1Acm))):])))
    # print("angle of 100Hz n1Acm:{}".format(agn))
    # print("angle of 100Hz :{}".format(agx))
    # print("amplitude of 100Hz noise:{}".format(apn))
    # if apn < 4:
    #     break
    
def testMirrorSnr():
    x0,n0 = getSig(t,F0,FP,FS,-60)
    win = 10000
    # plt.figure()
    x1 = getAcmSig(x0,win,win)
    n1 = getAcmSig(n0,win,win)
    x2 = getAcmSigEnhanced(x0,win,FS,F0)
    n2 = getAcmSigEnhanced(n0,win,FS,F0)
    
    x2 = (x2 - x2[np.arange(len(x2) - 1, -1, -1)]) / 2
    n2= (n2 - n2[np.arange(len(n2) - 1, -1, -1)]) / 2
    
    sig1 = x1 + n1 #getAcmSig(x1+n1,win,win)
    sig2 = x2 + n2 #getAcmSig(x2+n2,win,win)
    plt.subplot(2,2,1)
    showInF(sig1,FP,FS)
    plt.subplot(2,2,2)
    showInF(sig2,FP,FS)
    plt.subplot(2,2,3)
    showInF(n1,FP,FS)
    plt.subplot(2,2,4)
    showInF(n2,FP,FS)
    plt.show()
    print("snr(x1,n1):{}db".format(snr(x1,n1)))
    print("snr(x2,n2):{}db".format(snr(x2,n2)))
    print("gain:{}db".format(snr(x2,n2) - snr(x1,n1)))

def showFeature(st,sp,se,featureName):
    for snr in range(st,sp,se):
        with open("features\\test\\{}{}.pkl".format(snr,featureName),'rb') as f:
            X,y = pkl.load(f)
        X = X[:,-1]
        plt.figure()
        for i in range(len(y)):
            if y[i] == 0:
                plt.scatter(X[i],X[i],c = 'r')
            elif y[i] == 1:
                plt.scatter(X[i],X[i],c = 'b')
        plt.show()



def getSinglePointSig(sig,fS,f0):
    df = fS / len(sig)
    f = np.arange(len(sig)) / len(sig) * fS
    f02N = np.where(np.abs(f - f0) < df / 2)[0][0]
    sigF = np.fft.fft(sig)
    singlePointSigF = np.zeros(len(sigF),dtype=complex)
    singlePointSigF[f02N] = sigF[f02N]
    singlePointSigF[-f02N] = sigF[-f02N]

    singlePointSig = np.real(np.fft.ifft(singlePointSigF))
    return singlePointSig,singlePointSigF[f02N]


def sigOrNoise(sig,fS = FS,f0 = F0):
    
    sgpSig,sgpSigF0 = getSinglePointSig(sig,FS,F0)
    
    
    plt.plot(sgpSig)
    plt.xlim([0,200])
    plt.title('Amplitude:{:.2f},Phase:{:.2f}'.format(np.max(sgpSig),1 / np.pi * 180 * np.angle(sgpSigF0)))



# if __name__ == "__main__":
#     sigOrNoise()
#     # testFake()
#     # testMirrorSnr()
#     # testSTFT()
#     # st = -53
#     # sp = -54
#     # se = -1
#     # featureName = 'Fake'
#     # # dataGeneration(st,sp,se,featureName)
#     # testSinglePoint(st,sp,se,featureName)
#     # # showFeature(st,sp,se,featureName)