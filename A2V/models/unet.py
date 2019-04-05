import torch
import numpy as np
import scipy as sc
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# Identity Encoder and Frame Decoder
from .config import (NOISE_OUTPUT, HIDDEN_SIZE_NOISE, NUM_LAYERS_NOISE)

from .audio_encoder import AudioEncoder

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_x, kernel_y, stride, padding, cuda=False):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(kernel_x, kernel_y),
            stride=stride,
            padding=padding),
        nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, d_k_x, d_k_y, s_d, c_k_x, c_k_y, s_c, padding, debug=False):
        super(Up, self).__init__()

        self.debug = debug

        self.deconv = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(d_k_x, d_k_y),
            stride=s_d
            ),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
        nn.Conv2d(
            in_channels=2*out_ch,
            out_channels=out_ch,
            kernel_size=(c_k_x, c_k_y),
            stride=s_c,
            padding=padding
            ),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        if self.debug:
            print("x1 size ", x1.size())
            print("x2 size ", x2.size())
        x1 = self.deconv(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        if self.debug:
            print("x1 after deconv size ", x1.size())
            print("x2 after pad size ", x2.size())
        x = torch.cat([x2, x1], dim=1)

#         x = torch.add(x1, x2)
        if self.debug:
            print("x size ", x.size())
        x = self.conv(x)
        return x



class Unet(nn.Module):
    def __init__(self, debug=False):
        super(Unet, self).__init__()

        # Identity Encoder
        self.debug = debug

        self.i_c_1 = Down(3, 64, 4, 4, 2, 1)
        self.i_c_2 = Down(64, 128, 4, 4, 2, 1)
        self.i_c_3 = Down(128, 256, 4, 4, 2, 1)
        self.i_c_4 = Down(256, 512, 4, 4, 2, 1)
        self.i_c_5 = Down(512, 1024, 4, 4, 2, 1)
        self.i_c_6 = Down(1024, 50, 3, 3, 2, 0)

        # Frame decoder

        self.f_d_1 = Up(316, 1024, 3, 3, 1, 3, 3, 1, 0)
        self.f_d_2 = Up(1024, 512, 4, 4, 2, 3, 3, 1, 1)
        self.f_d_3 = Up(512, 256, 4, 4, 2, 3, 3, 1, 1)
        self.f_d_4 = Up(256, 128, 4, 4, 2, 3, 3, 1, 1)

        self.f_d_5 = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(4,4),
            stride=2,
            padding=0,
            ),
            nn.ReLU()
        )

        self.f_d_6 = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=64,
            out_channels=3,
            kernel_size=(4,4),
            stride=2,
            padding=0
            ),
            nn.ReLU()
        )

        self.audio_encoder = AudioEncoder()
        self.noise_encoder = nn.GRU(NOISE_OUTPUT, HIDDEN_SIZE_NOISE, NUM_LAYERS_NOISE)

    def forward(self, image_data, audio_data, noise_data):
        i_64 = self.i_c_1(image_data)
        if self.debug:
            print(i_64.size())
        i_128 = self.i_c_2(i_64)
        if self.debug:
            print(i_128.size())
        i_256 = self.i_c_3(i_128)
        if self.debug:
            print(i_256.size())
        i_512 = self.i_c_4(i_256)
        if self.debug:
            print(i_512.size())
        i_1024 = self.i_c_5(i_512)
        if self.debug:
            print(i_1024.size())

        # concatenate noise and audio
        latent = self.i_c_6(i_1024)
        if self.debug:
            print("Latent Encoder size ", latent.size())
        # Audio Data

        audio_encoded = self.audio_encoder(audio_data)
        audio_encoded = audio_encoded.view(audio_encoded.size()[0], audio_encoded.size()[2], 1, 1)
        if self.debug:
            print("Audio Encoded size ", audio_encoded.size())

        # Noise part
        output_noise, hn_noise = self.noise_encoder(noise_data)
        output_noise = output_noise.view(output_noise.size()[0], output_noise.size()[2], 1, 1)
        if self.debug:
            print("Noise output size ", output_noise.size())

        latent = torch.cat((audio_encoded, latent, output_noise), dim=1)
        if self.debug:
            print("New latent size ", latent.size())


        # print(latent.size())
        # print("---------------------")
        f_1024 = self.f_d_1(latent, i_1024)
        if self.debug:
            print('f 1024 ', f_1024.size())
        f_512 = self.f_d_2(f_1024, i_512)
        if self.debug:
            print('f 512 ', f_512.size())
        f_256 = self.f_d_3(f_512, i_256)
        if self.debug:
            print('f 256 ', f_256.size())
        f_128 = self.f_d_4(f_256, i_128)
        if self.debug:
            print('f 128 ', f_128.size())
        f_64 = self.f_d_5(f_128)
        if self.debug:
            print('f 64 ', f_64.size())
        f_3 = self.f_d_6(f_64)
        if self.debug:
            print('f 3 ', f_3.size())
        diffX = 96 - f_3.size()[2]
        diffY = 96 - f_3.size()[3]
        f_3 = F.pad(f_3, (diffX//2, int(diffX/2), diffY//2, int(diffY/2)))
        return f_3

# import matplotlib.pyplot as plt
# def visualize_image(inp):
#     """
#         See a particular image
#     """
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     plt.imshow(inp)
# #     plt.show()
#
# visualize_image(inputs[0])
# # # inputs[0].size()
# # print(inputs.size())
# # print(type(out[0].detach().numpy()))
# visualize_image(out[0].detach())
