from unittest.mock import inplace

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self,in_channels,out_channels,type):
        super(Conv_Block,self).__init__()
        self.type=type
        if type=='img':
            self.layer=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=False)
            )
        elif type=='text':
            self.layer=nn.Sequential(
                nn.Conv1d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
                nn.BatchNorm1d(out_channels),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=False)
            )

    def forward(self,x):
            return self.layer(x)

class Down_Sample(nn.Module):#最大值池化下采样
    def __init__(self):
        super(Down_Sample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        return self.layer(x)

class Up_Sample(nn.Module):
    def __init__(self,in_channel):
        super(Up_Sample, self).__init__()
        self.layer = nn.Conv2d(in_channel, in_channel//2, kernel_size=1, stride=1)

    def forward(self,x,pred_x):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,pred_x),dim=1)