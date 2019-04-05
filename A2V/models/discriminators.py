import torch
import numpy as np
import scipy as sc
from torchvision import transforms, datasets
from .audio_encoder import AudioEncoder, ImageEncoder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class Convolution(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel,
                stride=stride),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class FrameDiscriminator(nn.Module):
    def __init__(self):
        super(FrameDiscriminator, self).__init__()

        self.c1 = Convolution(6, 64, 4, 2)
        self.c2 = Convolution(64, 128, 4, 2)
        self.c3 = Convolution(128, 256, 4, 2)
        self.c4 = Convolution(256, 512, 4, 2)
        self.c5 = Convolution(512, 1024, 4, 2)
        self.c6 = nn.Linear(1024, 128)
        self.c7 = nn.Linear(128, 1)

    def forward(self, target, still_image):
        x = torch.cat((target, still_image), dim=1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = x.view(-1, 1024)
        # print("x size ", x.size())
        x = self.c6(x)
        out = self.c7(x)
        out = F.sigmoid(out)
        return out


class SequenceDiscriminator(nn.Module):
    def __init__(self):
        super(SequenceDiscriminator, self).__init__()
        self.audio_encoder = AudioEncoder()
        self.image_encoder = ImageEncoder()
        self.gru_image = nn.GRU(50, 50, 2)
        self.gru_audio = nn.GRU(256, 256, 2)
        self.fc1 = nn.Linear(306, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, i_i, i_a):
        o_i = self.image_encoder(i_i)
        # print("o_i ", o_i.size())
        o_i = o_i.view(o_i.size()[0], 1, o_i.size()[1])
        o_i, h_n = self.gru_image(o_i)
        # print("o_i after GRU", o_i.size())
        # print("Audio Encoder")
        o_a = self.audio_encoder(i_a)
        # print("o_a ", o_a.size())
        # o_a = o_a.view(o_a.size()[0], 1, o_a.size())
        o_a, h_n = self.gru_audio(o_a)

        # print("o_a after GRU", o_a.size())
        x = torch.cat([o_a, o_i], dim=2)
        x = self.fc1(x)
        x = self.fc2(x)
        out = F.sigmoid(x)
        return out
