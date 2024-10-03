import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import networks
import torch
import torch.nn as nn

class pix2pix_loss(nn.Module):
    def __init__(self,l1_lambda=100,mode="vanilla"):
        super().__init__()
        self.ganloss=networks.GANLoss(gan_mode=mode)
        self.l1loss=torch.nn.L1Loss()
    def forward(self,x,label):
        if self.training:
            return (1-self.alpha)(1-self.ssim(x,label))+self.alpha*self.lloss(x,label)
        else:
            return (1-self.alpha)(1-self.ssim(x,label))+self.alpha*self.lloss(x,label), self.ssim,self.lloss