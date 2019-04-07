import cv2
import os
import glob
import shutil
import numpy as np
import scipy.io as sio
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


import math
import numpy as np
import numpy.matlib as matlab
import scipy.signal as signal
import scipy.fftpack as fft
from scipy.misc import imresize



def temporal_ce_loss(output, target, mask):

    seq_len=output.size()[1]
    target=target[:,:seq_len].contiguous()
    mask=mask[:,:seq_len].contiguous()

    # Using cross_entropy loss
    output=torch.log(output)

    # flatten all the labels and mask
    target = target.view(-1)

    mask=mask.view(-1)

    # flatten all predictions
    no_classes=output.shape[-1]
    output = output.view(-1,no_classes)

#     print(mask.shape, output.shape,target.shape)
#     print( output[:, target].shape)
    # count how many frames we have
    nb_frames = int(torch.sum(mask).item())

    # pick the values for the label and zero out the rest with the mask
#     output = output[:, target] * mask
    y=target.view(-1,1)
#     print(y.shape,torch.gather(output, 1, y).shape)
    output=torch.gather(output, 1, y)* mask

    # compute cross entropy loss which ignores all elem where mask =0
    ce_loss = -torch.sum(output) / nb_frames

    return ce_loss




def gen_lstm_batch_random(X, y, seqlen, batchsize=30, shuffle=True):
    """
    randomized data generator for training data
    creates an infinite loop of mini batches
    :param X: input
    :param y: target
    :param seqlen: lengths of video
    :param batchsize: number of videos per batch
    :return: x_train, y_target, input_mask, video idx used
    """
    # find the max len of all videos for creating the mask
    max_timesteps = np.max(seqlen)
    feature_len = X.shape[1]
    no_videos = len(seqlen)
    start_video = 0
    reset = False

    # compute integral lengths of the video for fast offset access for data matrix
    integral_lens = [0]
    for i in range(1, len(seqlen)):
        integral_lens.append(integral_lens[i-1] + seqlen[i - 1])

    # permutate the video sequences for each batch
    if shuffle:
        randomized = np.random.permutation(len(seqlen))
    else:
        randomized = range(len(seqlen))
    while True:
        end_video = start_video + batchsize
        if end_video >= no_videos:  # all videos iterated, reset
            batch_video_idxs = randomized[start_video:]
            # extract all the video lengths of the video idx
            reset = True
        else:
            batch_video_idxs = randomized[start_video:end_video]
        bsize = len(batch_video_idxs)
        X_batch = np.zeros((bsize, max_timesteps, feature_len), dtype=X.dtype)  # returned batch input
        y_batch = np.zeros((bsize,), dtype='uint8')

        vid_lens_batch = np.zeros((bsize,), dtype='uint8')
        mask = np.zeros((bsize, max_timesteps), dtype='uint8')

        # populate the batch X and batch y
        for i, idx in enumerate(batch_video_idxs):
            start = integral_lens[idx]
            l = seqlen[idx]
            end = start + l
            X_batch[i] = np.concatenate([X[start:end],
                                         np.zeros((max_timesteps - l, feature_len))])
            y_batch[i] = y[start]
            mask[i, :l] = 1  # set 1 for length of video
            mask[i, l:] = 0  # set 0 for rest of video
            vid_lens_batch[i]=l
        if reset:
            # permutate the new video sequences for each batch
            if shuffle:
                randomized = np.random.permutation(len(seqlen))
            else:
                randomized = range(len(seqlen))
            start_video = 0
            reset = False
        else:
            start_video = end_video


        yield X_batch, y_batch,vid_lens_batch, mask, batch_video_idxs



def compute_integral_len(lengths):
    # compute integral lengths of the video for fast offset access for data matrix
    integral_lens = [0]
    for i in range(1, len(lengths)):
        integral_lens.append(integral_lens[i - 1] + lengths[i - 1])
    return integral_lens


def gen_seq_batch_from_idx(data, idxs, seqlens, integral_lens, max_timesteps):
    feature_len = data.shape[-1]
    X_batch = np.zeros((len(idxs), max_timesteps, feature_len), dtype=data.dtype)

    for i, seq_id in enumerate(idxs):
        l = seqlens[seq_id]
        start = integral_lens[seq_id]
        end = start + l
        X_batch[i] = np.concatenate([data[start:end],
                                     np.zeros((max_timesteps - l, feature_len), dtype=data.dtype)])
    return X_batch




def gen_lstm_batch_random_test(X, y, seqlen, batchsize=30, shuffle=True):
    """
    randomized data generator for training data
    creates an infinite loop of mini batches
    :param X: input
    :param y: target
    :param seqlen: lengths of video
    :param batchsize: number of videos per batch
    :return: x_train, y_target, input_mask, video idx used
    """
    # find the max len of all videos for creating the mask
    max_timesteps = np.max(seqlen)
    feature_len = X.shape[1]
    no_videos = len(seqlen)
    start_video = 0
    reset = False
    reset_g=False

    # compute integral lengths of the video for fast offset access for data matrix
    integral_lens = [0]
    for i in range(1, len(seqlen)):
        integral_lens.append(integral_lens[i-1] + seqlen[i - 1])

    # permutate the video sequences for each batch
    if shuffle:
        randomized = np.random.permutation(len(seqlen))
    else:
        randomized = range(len(seqlen))
    while True:
        if reset_g==True:
            reset_g=False
            yield 0


        end_video = start_video + batchsize
        if end_video >= no_videos:  # all videos iterated, reset
            batch_video_idxs = randomized[start_video:]
            # extract all the video lengths of the video idx
            reset = True
        else:
            batch_video_idxs = randomized[start_video:end_video]
        bsize = len(batch_video_idxs)
        X_batch = np.zeros((bsize, max_timesteps, feature_len), dtype=X.dtype)  # returned batch input
        y_batch = np.zeros((bsize,), dtype='uint8')

        vid_lens_batch = np.zeros((bsize,), dtype='uint8')
        mask = np.zeros((bsize, max_timesteps), dtype='uint8')

        # populate the batch X and batch y
        for i, idx in enumerate(batch_video_idxs):
            start = integral_lens[idx]
            l = seqlen[idx]
            end = start + l
            X_batch[i] = np.concatenate([X[start:end],
                                         np.zeros((max_timesteps - l, feature_len))])
            y_batch[i] = y[start]
            mask[i, :l] = 1  # set 1 for length of video
            mask[i, l:] = 0  # set 0 for rest of video
            vid_lens_batch[i]=l
        if reset:
            # permutate the new video sequences for each batch
            if shuffle:
                randomized = np.random.permutation(len(seqlen))
            else:
                randomized = range(len(seqlen))
            start_video = 0
            reset = False
            reset_g=True
        else:
            start_video = end_video


        yield X_batch, y_batch,vid_lens_batch, mask, batch_video_idxs




