import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio

from config import (SEQ_LEN, AUDIO_OUTPUT, BATCH, HIDDEN_SIZE_AUDIO, NUM_LAYERS_AUDIO, AUDIO_PATH)

class Conv1DAudio(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super(Conv1DAudio, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.a_c_1 = Conv1DAudio(1, 16, 250, 50, 100)
        self.a_c_2 = Conv1DAudio(16, 32, 4, 2, 1)
        self.a_c_3 = Conv1DAudio(32, 64, 4, 2, 1)
        self.a_c_4 = Conv1DAudio(64, 128, 4, 2, 1)
        self.a_c_5 = Conv1DAudio(128, 256, 4, 2, 1)
        self.a_c_6 = Conv1DAudio(256, 512, 4, 2, 1)
        self.fc1 = nn.Linear(2*512, 256)
        self.gru = nn.GRU(AUDIO_OUTPUT, HIDDEN_SIZE_AUDIO, NUM_LAYERS_AUDIO)
        self.act = nn.Tanh()

    def forward(self, x):
        ac1 = self.a_c_1(x)
        ac2 = self.a_c_2(ac1)
        ac3 = self.a_c_3(ac2)
        ac4 = self.a_c_4(ac3)
        ac5 = self.a_c_5(ac4)
        ac6 = self.a_c_6(ac5)
        ac6 = ac6.view(-1, 2*512)
        out = self.fc1(ac6)
        out = out.view(out.size()[0] ,1, out.size()[1])
        rnn = nn.GRU(AUDIO_OUTPUT, HIDDEN_SIZE_AUDIO, NUM_LAYERS_AUDIO)
        output, hn = self.gru(out)
        output = self.act(output)
        return output

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

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.i_c_1 = Convolution(3, 64, 3, 2)
        self.i_c_2 = Convolution(64, 128, 3, 2)
        self.i_c_3 = Convolution(128, 256, 3, 2)
        self.i_c_4 = Convolution(256, 512, 3, 2)
#         self.i_c_5 = Convolution(512, 1024, 4, 2)
        self.i_c_5 = Convolution(512, 256, 3, 1)
        self.i_c_6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=50,
                kernel_size=(3,3),
                stride=2),
            nn.ReLU()
        )
        self.tanh = nn.Tanh()

    def forward(self, x):

        i_64 = self.i_c_1(x)
        # print(i_64.size())
        i_128 = self.i_c_2(i_64)
        # print(i_128.size())
        i_256 = self.i_c_3(i_128)
        # print(i_256.size())
        i_512 = self.i_c_4(i_256)
        # print(i_512.size())
        i_1024 = self.i_c_5(i_512)
        # print(i_1024.size())
        latent = self.i_c_6(i_1024)
        return self.tanh(latent)
