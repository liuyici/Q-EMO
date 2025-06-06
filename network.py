import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import Adver_network
from typing import Optional, Tuple
import torch.optim
from   torch                            import autograd
from   torch.autograd                   import Variable
from   core_qnn.quaternion_layers       import *
import sys

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r = input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight), r * torch.tanh(self.bias))
        output = cosine
        return output

# ***********
# Our modules
# ***********



def GaussianNoise(x, sigma = 1.0):
    noise = torch.tensor(0.0).cuda()
    sampled_noise = noise.repeat(*x.size()).normal_(mean=0, std=sigma)
    x = x + sampled_noise
    return x

  

class DFN(nn.Module):
    def __init__(self, input_size = 2790, hidden_size = 320, use_bottleneck=True, bottleneck_dim=256, radius=10.0, class_num=3):
        super(DFN, self).__init__()
        self.feature_extractor = nn.Sequential(
            QuaternionLinear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.1, affine=False),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Flatten()
        )
        self.layer1 = nn.Sequential(
            QuaternionLinear(input_size, hidden_size),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.layer2 = nn.Sequential(
            QuaternionLinear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Flatten(),
        )
    
        self.bottleneck = nn.Sequential(
            QuaternionLinear(hidden_size, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.__in_features = bottleneck_dim
        self.radius = radius
        self.fc = SLR_layer(bottleneck_dim, class_num)
    def forward(self, x):
        ####### update #############
        return x, y
    
    def output_num(self):
      return self.__in_features
  
    def get_parameters(self):
        # return the parameters of the deep neural network
        parameter_list = [{"params": self.feature_extractor.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                          {"params": self.bottleneck.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                          {"params": self.fc.parameters(), "lr_mult": 1, 'decay_mult': 2}]
  
        return parameter_list


class DiscriminatorDANN(nn.Module):
    def __init__(self, in_feature, hidden_size, radius=10.0, max_iter=10000):
        super(DiscriminatorDANN, self).__init__()

        self.radius = radius
        self.ad_layer1 = nn.Linear(in_feature, hidden_size + 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)

        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(hidden_size + 1, hidden_size + 1)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)


        self.ad_layer3 = nn.Linear(hidden_size + 1, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)

        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3, nn.Sigmoid())



    def forward(self, x, y=None):
        f2 = self.fc1(x)
        f = self.fc2_3(f2)

        return f

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]

class Decoder(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 out_dim: int,
                 use_bottleneck=True,
                 radius=10.0, 
                 t: Optional[float] = 0.1):
        # self, input_size, hidden_size, use_bottleneck=True, bottleneck_dim=100, radius=10.0, class_num=1000
        super(Decoder, self).__init__()
            # set
        self.use_bottleneck = use_bottleneck
    
        self.bottleneck_fc = QuaternionLinear(hidden_size, out_dim)
        self.bottleneck = nn.Sequential(
            QuaternionLinear(hidden_size, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # self.bottleneck_fc.apply(init_weights)
        self.__in_features = out_dim

        self.radius = radius
        # self.radius = nn.Parameter(torch.tensor(radius, requires_grad=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck(x)
        x = self.radius * x / (torch.norm(x, dim=1, keepdim=True) + 1e-10)
        return x
    def get_parameters(self):
       # return the parameters of the deep neural network
    #    parameter_list = [{"params": self.bottleneck.parameters(), "lr_mult": 1, 'decay_mult': 2}]
       return  [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]   
