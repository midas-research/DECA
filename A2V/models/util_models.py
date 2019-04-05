import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D(nn.Module):
    def __init__(self, in, out, kernel_x, kernel_y, stride):
        super(Conv1D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in, out, kernel_size=(kernel_x, kernel_y), stride=stride),
            nn.BatchNorm2d(out),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out

"""
U-net models
"""

class ConvDown2D(nn.Module):
    def __init__(self, in, out, kernel_x, kernel_y, stride):
        super(ConvDown2D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in,
                out_channels=out,
                kernel_size=(kernel_x, kernel_y),
                stride=stride
            ),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DeconvUp2D(nn.Module):

    def __init__(self, in, out, d_k_x, d_k_y, s_d, c_k_x, c_k_y, s_c):
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in,
                out_channels=out,
                kernel_size=(d_k_x, d_k_y),
                stride=s_d
            ),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in,
                out_channels=out,
                kernel_size=(c_k_x, c_k_y),
                stride=s_c
            )
        )

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2) ))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
