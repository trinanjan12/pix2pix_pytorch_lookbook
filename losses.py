import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#============================================
# GAN loss
#============================================


class LSGANLoss(nn.Module):
    def __init__(self, device):
        super(LSGANLoss,self).__init__()
        self.device = device
        self.loss_fn = nn.MSELoss()
        return

    def forward_D(self, d_real, d_fake):
        real_ones_tsr = torch.ones(d_real.shape).to(self.device)
        fake_zeros_tsr = torch.zeros(d_fake.shape).to(self.device)
        loss_D_real = self.loss_fn(d_real, real_ones_tsr)
        loss_D_fake = self.loss_fn(d_fake, fake_zeros_tsr)
        loss_D = (loss_D_real + loss_D_fake) * .5
        return loss_D, loss_D_real, loss_D_fake

    def forward_G(self, d_fake):
        real_ones_tsr = torch.ones(d_fake.shape).to(self.device)
        loss_G = self.loss_fn(d_fake, real_ones_tsr)
        return loss_G

    def forward(self, d_real, d_fake, dis_or_gen=True):
        # Loss for Discriminator
        if dis_or_gen:
            loss, _, _ = self.forward_D(d_real, d_fake)
        # Loss for Generator
        else:
            loss = self.forward_G(d_fake)

        return