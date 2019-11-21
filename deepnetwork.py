import torch
from torch import nn
from torchvision import models


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet,self).__init__()

        # vgg = models.vgg19(pretrained=True).cuda()
        # self.feature = vgg
        # self.feature.add_module('global average',nn.AvgPool2d(9))
        self.feature = nn.Sequential(
            nn.Conv2d(1,3,5),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2,2),
            nn.Conv2d(3,3,5),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2,2),
            nn.Conv2d(3,3,5),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2,2),
        )
        self.classfier = nn.Sequential(
            nn.Linear(8 * 8 * 3,100),
            nn.ReLU(),
            nn.Linear(100,2),
            nn.Softmax()
        )
        self.crit = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(),lr = 0.001)

    def fit(self,X,y):
        batch_size = 50
        X = torch.Tensor(X).cuda()
        y = torch.Tensor(y).cuda()
        dataset = torch.utils.data.TensorDataset(X,y)
        loader = torch.utils.data.DataLoader(dataset,batch_size,True)
        for epoch in range(400):
            for i,data in enumerate(loader):
                input,label = data

                self.opt.zero_grad()

                ft = self.feature(input).view(input.size(0),-1)
                otpt = self.classfier(ft)
                loss = self.crit(otpt,label.long())
                loss.backward()

                self.opt.step()
                if epoch % 50 == 0:
                    print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item()))
                


    def forward(self,x):
        x = torch.from_numpy(x).float().cuda()
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        otpt = self.classfier(x)
        _,pred = torch.max(otpt,1)
        return pred
