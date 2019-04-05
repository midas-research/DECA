import torch
import numpy as np
import scipy as sc
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

SEQ_LEN = 1
NOISE_OUTPUT = 10
BATCH = 38
HIDDEN_SIZE_NOISE = 10
NUM_LAYERS_NOISE = 1
noise = torch.rand(BATCH, 1, NOISE_OUTPUT)
noise = Variable(noise)
noise = noise.data.resize_(noise.size()).normal_(0, 0.6)
rnn = nn.GRU(NOISE_OUTPUT, HIDDEN_SIZE_NOISE, NUM_LAYERS_NOISE)
output_noise, hn_noise = rnn(noise)
