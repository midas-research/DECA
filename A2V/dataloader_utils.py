import os
import torch
#import torchaudio
import numpy as np
import torch.nn as nn
from torchvision import transforms, datasets
import torch.utils.data as utils
from torch.autograd import Variable

from config import (AUDIO_PATH, FRAMES_PATH, AUDIO_DATA_PATH, VIDEO_DATA_PATH, VIDEO_TEST_PATH)

def create_overlapping_samples(sound):
    """
    function: to convert an audio into overlapping continous samples
    """

    sound_avg = sound.mean(1)
    audio_len = sound_avg.size()[0]
    sample_window = 8000
    overlapping = 2000
    left = 0
    right = 8000
    audio_data = torch.Tensor()
    while left < audio_len:
        sample = sound_avg[left:right]
        if sample.size()[0] != 8000:
            break
        sample = sample.view(1, 1, sample.size()[0])
        audio_data = torch.cat((audio_data, sample))
        left = right - overlapping
        right = left + sample_window

    return audio_data

def get_audio_data():
    # get all audios
    audios = os.listdir(AUDIO_PATH)
    print(audios)
    # sound, sample_rate = torchaudio.load(AUDIO_PATH + audios[0])

    # create overlapping samples
    batch = create_overlapping_samples(sound)
    return batch

def get_image_data():

    data_transform = {
            'frames': transforms.Compose([
                transforms.RandomCrop(96),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    frames = datasets.ImageFolder(FRAMES_PATH, data_transform['frames'])
    dset_loader = torch.utils.data.DataLoader(frames, batch_size=38, shuffle=False, num_workers=4)
    # model = Unet()
    # out = model()
    for data in dset_loader:
        inputs, _ = data
    return inputs
    # print(inputs.size())
    # image = torch.rand(1, 3, 96, 96)
    # out = model(image)
    # print(out.size())

def get_data(view, utterences):
    audio_batch = []
    video_batch = []
    # audio_files = sorted(os.listdir(AUDIO_DATA_PATH))
    video_files = sorted(os.listdir(VIDEO_DATA_PATH))
    for file_name in video_files:
        try:
            temp = file_name.split(".")[0]
            ind = temp.index("u") + 1
            u = int(temp[ind:])
            if (view in file_name) and u <= utterences:
                video_batch.append(file_name)
                audio_batch.append(file_name)
        except:
            print("SKIPPING: Something is wrong with {} file".format(file_name))
    print("Number of files prepared ", len(video_batch))
    return (video_batch, audio_batch)

def get_test_data(view, lower, upper):
    audio_batch = []
    video_batch = []
    # audio_files = sorted(os.listdir(AUDIO_TEST_PATH))
    video_files = sorted(os.listdir(VIDEO_TEST_PATH))
    #print("num files ", len(video_files))
    for file_name in video_files:
        try:
            temp = file_name.split(".")[0]
            ind = temp.index("u") + 1
            u = int(temp[ind:])
            #print("till here")
            #print(view, file_name, u, lower, upper)
            if (view in file_name) and (u > lower) and (u <= upper):
                video_batch.append(file_name)
                audio_batch.append(file_name)
        except:
            print("SKIPPING: Something is wrong with {} file".format(file_name))
    print("Number of files prepared ", len(video_batch))
    return (video_batch, audio_batch)
