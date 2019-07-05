import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

import torchvision
import pretrainedmodels




OUTPUT_DIM = {
    'alexnet'       :  256,
    'vgg11'         :  512,
    'vgg13'         :  512,
    'vgg16'         :  512,
    'vgg19'         :  512,
    'resnet18'      :  512,
    'resnet34'      :  512,
    'resnet50'      : 2048,
    'resnet101'     : 2048,
    'resnet152'     : 2048,
    'densenet121'   : 1024,
    'densenet161'   : 2208,
    'densenet169'   : 1664,
    'densenet201'   : 1920,
    'squeezenet1_0' :  512,
    'squeezenet1_1' :  512,
    'resnext101_64x4d' :  2048,
    'nasnetalarge' :  2048,
    'se_resnet101' :  2048,
}

def ramac(x, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension
    
    W = x.size(3)
    H = x.size(2)
    
    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)
    
    b = (max(H, W)-w)/(steps-1)
    (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension
    
    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    #print(idx.tolist())
    if H < W:
        Wd = idx.tolist()#[0]
    elif H > W:
        Hd = idx.tolist()#[0]

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # find attention
    tt=(x.sum(1)-x.sum(1).mean()>0)
    # caculate weight
    weight=tt.sum().float()/tt.size(-2)/tt.size(-1)
    # ingore
    if weight.data<=1/3.0:
        weight=weight-weight

    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v) * weight

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)
    
        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
        
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:,:,(int(i_)+torch.Tensor(range(int(wl))).long()).tolist(),:]
                R = R[:,:,:,(int(j_)+torch.Tensor(range(int(wl))).long()).tolist()]
                # obtain map
                tt=(x.sum(1)-x.sum(1).mean()>0)[:,(int(i_)+torch.Tensor(range(int(wl))).long()).tolist(),:][:,:,(int(j_)+torch.Tensor(range(int(wl))).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                # caculate each region
                weight=tt.sum().float()/tt.size(-2)/tt.size(-1)
                if weight.data<=1/3.0:
                    weight=weight-weight
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt) * weight
                v += vt

    return v
	
class RAMAC(nn.Module):
    
    def __init__(self, L=3, eps=1e-6):
        super(RAMAC,self).__init__()
        self.L = L
        self.eps = eps
    
    def forward(self, x):
        return ramac(x, L=self.L, eps=self.eps)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'

class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # features -> pool -> norm
        x = self.features(x)
        feature_RAMAC = self.norm(RAMAC()(x)).squeeze(-1).squeeze(-1)

        return feature_RAMAC.permute(1,0)


def init_network(model='vgg16'):

    
    net_in = pretrainedmodels.__dict__[model](num_classes=1000, pretrained='imagenet')
    mean=net_in.mean
    std=net_in.std

    if model.startswith('vgg'):
        net_in = getattr(torchvision.models, model)(pretrained=True)
        features = list(list(net_in.children())[0][:-1])
    elif model.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif model.startswith('resnext101_64x4d'):
        features = list(net_in.children())[:-2]
    elif model.startswith('se'):
        features = list(net_in.children())[:-2]
    else:
        raise ValueError('Unknown model: {}!'.format(model))
    
    dim = OUTPUT_DIM[model]

    # create meta information to be stored in the network
    meta = {'architecture':model, 'outputdim':dim, 'mean':mean, 'std':std}

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, meta)

    return net


def extract_feature(images, model_choose, image_size):
    
    model = init_network(model=model_choose)
    vecs_RAMAC, name_list = extract_vectors(model, images, image_size, print_freq=100)
    
    feats_RAMAC = vecs_RAMAC.numpy()

    return feats_RAMAC