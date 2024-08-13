#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut

        self.weight1 = torch.nn.Parameter(torch.FloatTensor(1619, nOut), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.FloatTensor(10, nOut), requires_grad=True)

        self.ce = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.weight1, gain=1)
        nn.init.xavier_normal_(self.weight2, gain=1)

        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x1,x2, label1=None,label2=None):

        assert x1.size()[0] == label1.size()[0]
        assert x1.size()[1] == self.in_feats
        
        # cos(theta)
        cosine1 = F.linear(F.normalize(x1), F.normalize(self.weight1))
        # cos(theta + m)
        sine1 = torch.sqrt((1.0 - torch.mul(cosine1, cosine1)).clamp(0, 1))
        phi1 = cosine1 * self.cos_m - sine1 * self.sin_m

        if self.easy_margin:
            phi1 = torch.where(cosine1 > 0, phi1, cosine1)
        else:
            phi1 = torch.where((cosine1 - self.th) > 0, phi1, cosine1 - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot1 = torch.zeros_like(cosine1)
        one_hot1.scatter_(1, label1.view(-1, 1), 1)
        output1 = (one_hot1 * phi1) + ((1.0 - one_hot1) * cosine1)
        output1 = output1 * self.s

        loss1    = self.ce(output1, label1)
        prec1   = accuracy(output1.detach(), label1.detach(), topk=(1,))[0]



        # cos(theta)
        cosine2 = F.linear(F.normalize(x2), F.normalize(self.weight2))
        # cos(theta + m)
        sine2 = torch.sqrt((1.0 - torch.mul(cosine2, cosine2)).clamp(0, 1))
        phi2 = cosine2 * self.cos_m - sine2 * self.sin_m

        if self.easy_margin:
            phi2 = torch.where(cosine2 > 0, phi2, cosine2)
        else:
            phi2 = torch.where((cosine2 - self.th) > 0, phi2, cosine2 - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot2 = torch.zeros_like(cosine2)
        one_hot2.scatter_(1, label2.view(-1, 1), 1)
        output2 = (one_hot2 * phi2) + ((1.0 - one_hot2) * cosine2)
        output2 = output2 * self.s

        loss2    = self.ce(output2, label2)
        prec2   = accuracy(output2.detach(), label2.detach(), topk=(1,))[0]


        return loss1,loss2, prec1, prec2
