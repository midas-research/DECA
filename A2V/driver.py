import sys
import torch
import numpy as np
import scipy as sc
from torchvision import transforms, datasets
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import time
from models.audio_encoder import AudioEncoder
from models.unet import Unet
from models.discriminators import FrameDiscriminator, SequenceDiscriminator
from dataloader_utils import get_audio_data, get_image_data, get_data, get_test_data
from config import (SEQ_LEN, AUDIO_OUTPUT, BATCH, HIDDEN_SIZE_AUDIO, NUM_LAYERS_AUDIO, AUDIO_PATH, NOISE_OUTPUT, learning_rate)
from config import (AUDIO_PATH, FRAMES_PATH, AUDIO_DATA_PATH, VIDEO_DATA_PATH, VIDEO_TEST_PATH)
from PIL import Image
#import matplotlib.pyplot as plt
#import cv2
import random
import time
#sys.exit()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

EPS = 1e-06
def show_image(image):
    # If dimension is 3 x m x n
    print(image.shape)
    image = image.transpose(1, 2, 0)
    # image = image/float(255)
    print(image.shape)
    print("TILL HERE")
    # plt.imsave('img.jpg', image)
    plt.imshow(image)
    plt.show()

def dis_loss(FDwG1, FDwO1, SDwG2, SDwO2 ):

    return (
        torch.mean(torch.log(EPS + FDwO1)) +
        torch.mean(torch.log(EPS + 1 - FDwG1)) +
        torch.mean(torch.log(EPS + SDwO2)) +
        torch.mean(torch.log(EPS + 1 - SDwG2))
    )

def fdis_loss(FDwG1, FDwO1):
    return (
        torch.mean(torch.log(EPS + FDwO1)) +
        torch.mean(torch.log(EPS + 1 - FDwG1))
        )

def sdis_loss(SDwG2, SDwO2):
    return (
        torch.mean(torch.log(EPS + SDwO2)) +
        torch.mean(torch.log(EPS + 1 - SDwG2))
        )

def gen_loss(FDwG1, SDwG2):
    return (
        torch.mean(torch.log(EPS + FDwG1)) +
        torch.mean(torch.log(EPS + SDwG2))
        )

def normalize(img):
    img = (img - img.min())/(img.max() - img.min())
    return img

def train(audios, videos, unet, frame_discriminator, sequence_discriminator):
    optimizer_unet = torch.optim.Adam(unet.parameters(), lr=0.0008)
    optimizer_fd = torch.optim.Adam(frame_discriminator.parameters(), lr=0.001)
    optimizer_sd = torch.optim.Adam(sequence_discriminator.parameters(), lr=0.001)

    tick = time.time()
    num_epochs = 150
    best_loss = 1000
    for epoch in range(num_epochs):
        batch_g_loss = 0
        batch_d_loss = 0
        rejected = 0
        for i in range(len(videos)):
            try:
                video_d = np.load(os.path.join(VIDEO_DATA_PATH,videos[i]))
                # print("YES LOADED")
                # print(video_d.shape)
                video_d = video_d.transpose(0, 3, 1, 2)
                for j in range(video_d.shape[0]):
                    video_d[j] = (video_d[j] - video_d[j].min())/(video_d[j].max() - video_d[j].min())
                video_d = torch.from_numpy(video_d)

                #print(video_d.shape)
                video_data = Variable(video_d)  # this needs to be array of still frames

                audio_d = np.load(os.path.join(AUDIO_DATA_PATH,audios[i]))
                audio_d = torch.from_numpy(audio_d)
                audio_d = audio_d.view(audio_d.size()[0], audio_d.size()[2], audio_d.size()[1])
                audio_data = Variable(audio_d)

                MINIBATCHSIZE = video_d.size()[0]

                still_frame = video_d[2]
                # To show a still frame
                # show_image(still_frame.numpy())

                still_frame = still_frame.view(1, still_frame.size()[0], still_frame.size()[1], still_frame.size()[2])
                still_frame = Variable(still_frame.repeat(MINIBATCHSIZE, 1, 1, 1))
                noise_data = Variable(torch.rand(MINIBATCHSIZE, 1, NOISE_OUTPUT))
                # print(noise_data.size())
                noise_data = noise_data.data.resize_(noise_data.size()).normal_(0, 0.6)
                # plt.imshow(noise_data.numpy())
                # plt.show()
                # print(noise_data)
                # print(noise_data.size())

                if cuda:
                    audio_data = audio_data.cuda()
                    video_data = video_data.cuda()
                    noise_data = noise_data.cuda()
                    still_frame = still_frame.cuda()

                # Train Generator
                # print(audio_data.size())
                # print(video_data.size())
                # print(still_frame.size())
                optimizer_unet.zero_grad()
                optimizer_fd.zero_grad()
                optimizer_sd.zero_grad()

                gen_frames = unet(still_frame, audio_data, noise_data)
                # print(gen_frames[0])

                # img = gen_frames[random.randint(0, MINIBATCHSIZE -1)].cpu().detach().numpy().transpose(1, 2, 0)
                # cv2.imwrite('./logs/'+str(time.time())+'.jpg', img)


                #print(gen_frames.size())
                #img = gen_frames[0].detach().numpy()
                #show_image(img)
                #return
                # print("Generated frames ", gen_frames.size())

                Lambda = 100
                # print(video_data[0])
                # print(torch.mean(torch.mean(torch.mean(torch.abs(video_data - gen_frames), 1), 1), 1))

                #print(torch.mean(torch.mean(torch.mean(torch.mean(torch.abs(video_data - gen_frames), 1), 1), 1)))
                # return
                l1_loss = torch.mean(torch.mean(torch.mean(torch.mean(torch.abs(video_data - gen_frames), 1), 1), 1))
                #print(l1_loss)
                # return


                out1 = frame_discriminator(video_data, still_frame)
                out2 = frame_discriminator(gen_frames, still_frame)

                out3 = sequence_discriminator(video_data, audio_data)
                out4 = sequence_discriminator(gen_frames, audio_data)
                # print(out1)
                # print("out is ", out3.size())

                d_loss = -dis_loss(out2, out1, out4 ,out3)
                # frame_loss = fdis_loss(out2, out1)
                # sequence_loss = sdis_loss(out4, out3)
                # d_loss = -(frame_loss + sequence_loss)

                g_loss = -gen_loss(out2, out4) + Lambda*l1_loss
                g_loss.backward(retain_graph=True)
                optimizer_unet.step()

                d_loss.backward()
                optimizer_fd.step()
                optimizer_sd.step()


                #print("G loss {} FD loss {} SD loss {} D loss {}".format(g_loss.data, frame_loss.data, sequence_loss.data, d_loss.data))
                #print("G loss {}D loss {}".format(g_loss.data, d_loss.data))
                batch_g_loss += g_loss.data
                batch_d_loss += d_loss.data
                tock = time.time()
                # print("Epoch {}, Done for file: {}  Total time elapsed {} hr".format(epoch, videos[i], (tock-tick)/(60*60)))
            except Exception as e:
                rejected += 1
                print(e)
                # print("Something went wrong with the file {}".format(videos[i]))
        average_batch_g_loss = batch_g_loss/(len(videos) - rejected)
        average_batch_d_loss = batch_d_loss/(len(videos) - rejected)

        if epoch%10 == 0:
            torch.save(unet, view+str(epoch)+'Unet.pt')

        # generate_test_images(epoch, unet)
        if average_batch_g_loss < best_loss:
            torch.save(unet, view+'Unet.pt')
            #torch.save(frame_discriminator, view+'FrameDiscriminator.pt')
            #torch.save(sequence_discriminator, view+'SequenceDiscriminator.pt')
            #state = {
            #    'epoch': epoch,
            #    'unet': unet.state_dict(),
            #    'frame_discriminator': frame_discriminator.state_dict(),
            #    'sequence_discriminator': sequence_discriminator.state_dict(),
            #    'optimizer_unet': optimizer_unet.state_dict(),
            #    'optimizer_fd': optimizer_fd.state_dict(),
            #    'optimizer_sd': optimizer_sd.state_dict()
            #}
            #torch.save(state, view+'models_state.pt')

            best_loss = average_batch_g_loss
        print("E: {} G Loss: {} D loss {} R files: {}".format(epoch, average_batch_g_loss, average_batch_d_loss, rejected))

def generate_test_images(batch_no, unet):

    videos, audios = get_test_data()
    for i in range(len(videos)):
        save_path = './saved/'+str(batch_no)+'/'+str(i)+'/'
        os.makedirs(save_path)

        video_d = np.load(os.path.join(VIDEO_TEST_PATH,videos[i]))
        audio_d = np.load(os.path.join(AUDIO_TEST_PATH,audios[i]))
        video_d = torch.from_numpy(video_d)
        audio_d = torch.from_numpy(audio_d)
        print(audio_d.size())
        audio_d = audio_d.view(audio_d.size()[0], audio_d.size()[2], audio_d.size()[1])

        MINIBATCHSIZE = video_d.size()[0]

        audio_data = Variable(audio_d)
        video_data = Variable(video_d)  # this needs to be array of still frames
        still_frame = video_d[0]
        still_frame = still_frame.view(1, still_frame.size()[0], still_frame.size()[1], still_frame.size()[2])
        still_frame = Variable(still_frame.repeat(MINIBATCHSIZE, 1, 1, 1))
        noise_data = Variable(torch.rand(MINIBATCHSIZE, 1, NOISE_OUTPUT))
        noise_data = noise_data.data.resize_(noise_data.size()).normal_(0, 0.6)

        if cuda:
            audio_data = audio_data.cuda()
            video_data = video_data.cuda()
            noise_data = noise_data.cuda()
            still_frame = still_frame.cuda()

        gen_frames = unet(still_frame, audio_data, noise_data)
        for j in random.sample(range(0, MINIBATCHSIZE-1), 8):
            if cuda:
                img = gen_frames[j].cpu().detach().numpy().transpose(1, 2, 0)
            else:
                img = gen_frames[j].detach().numpy().transpose(1, 2, 0)
            cv2.imwrite(save_path+str(j)+'.png', img)
        # save_image(gen_frames, save_path+videos[i]+'.png')

view = '/mnt/data/rajivratn/lipsync/saved_models/v1_digits_lr_0.000003'
if not os.path.exists(view):
    os.makedirs(view)
def main():

    videos, audios = get_data("v1", 1)
    print("Data Loaded")
    # if os.path.exists('./Unet.pt'):
    #    unet = torch.load('./Unet.pt')
      #  frame_discriminator = torch.load()
    unet = Unet(debug=False)
    frame_discriminator = FrameDiscriminator()
    sequence_discriminator = SequenceDiscriminator()
    if cuda:
        print('SAHI JA RHA.......')
        unet = unet.cuda()
        frame_discriminator = frame_discriminator.cuda()
        sequence_discriminator = sequence_discriminator.cuda()
    # if torch.cuda.device_count() > 1:
        # print("Using ", torch.cuda.device_count(), " GPUs!")
        # unet = nn.DataParallel(unet)
        # frame_discriminator = nn.DataParallel(frame_discriminator)
        # sequence_discriminator = nn.DataParallel(sequence_discriminator)
    train(audios, videos, unet, frame_discriminator, sequence_discriminator)

main()


def test():
    batch_no = 0
    if os.path.exists('./4/30Unet.pt'):
       unet = torch.load('./4/30Unet.pt')

    videos, audios = get_test_data()
    for i in range(len(videos)):
        save_path = './saved_4/'+str(batch_no)+'/'+str(i)+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        video_d = np.load(os.path.join(VIDEO_TEST_PATH,videos[i]))
        video_d = video_d.transpose(0, 3, 1, 2)
        audio_d = np.load(os.path.join(AUDIO_DATA_PATH,audios[i]))

        stats = {
            'min': [],
            'max': []
        }

        for j in range(video_d.shape[0]):
            stats['min'].append(video_d[j].min())
            stats['max'].append(video_d[j].max())
            video_d[j] = (video_d[j] - video_d[j].min())/(video_d[j].max() - video_d[j].min())

        video_d = torch.from_numpy(video_d)
        audio_d = torch.from_numpy(audio_d)
        print(audio_d.size())
        audio_d = audio_d.view(audio_d.size()[0], audio_d.size()[2], audio_d.size()[1])

        MINIBATCHSIZE = video_d.size()[0]

        audio_data = Variable(audio_d)
        video_data = Variable(video_d)  # this needs to be array of still frames
        still_frame = video_d[0]
        still_frame = still_frame.view(1, still_frame.size()[0], still_frame.size()[1], still_frame.size()[2])
        still_frame = Variable(still_frame.repeat(MINIBATCHSIZE, 1, 1, 1))
        noise_data = Variable(torch.rand(MINIBATCHSIZE, 1, NOISE_OUTPUT))
        noise_data = noise_data.data.resize_(noise_data.size()).normal_(0, 0.6)

        if cuda:
            audio_data = audio_data.cuda()
            video_data = video_data.cuda()
            noise_data = noise_data.cuda()
            still_frame = still_frame.cuda()

        gen_frames = unet(still_frame, audio_data, noise_data)
        for j in random.sample(range(0, MINIBATCHSIZE-1), 4):
            if cuda:
                img = gen_frames[j].cpu().detach().numpy().transpose(1, 2, 0)
            else:
                img = gen_frames[j].detach().numpy().transpose(1, 2, 0)

            # Un normalize part will come here
            img = img*(stats['max'][j] - stats['min'][j]) + stats['min'][j]
            cv2.imwrite(save_path+str(j)+'.png', img)
        # save_image(gen_frames, save_path+videos[i]+'.png')



# get_data()


#test()
        # Save pictures in into video code will go here using openCV
saved_models_path = '/mnt/data/rajivratn/lipsync/saved_models/'
saved_video_path = '/mnt/data/rajivratn/lipsync/saved_videos/'
def prep_video_gen_data(view,lower, upper):
    print("Yes im called")
    model_path = saved_models_path + view + '_full_phrases/Unet.pt'
    video_path = saved_video_path + view + '/'
    if os.path.exists(model_path):
        unet = torch.load(model_path.cuda())
        #unet = torch.load(model_path, map_location=torch.device('cpu'))
        #
        unet = torch.nn.DataParallel(unet.cuda())
        print(unet)


    videos, audios = get_test_data(view, lower, upper)
    print("len of videos ", len(videos))
    for i in range(len(videos)):
        generate_video(VIDEO_TEST_PATH, videos[i], video_path, unet)


def generate_video(DATA_PATH, name, SAVE_PATH, unet):
    print("Genrating ", name)
    SAVE_PATH += name + '/'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    video_d = np.load(os.path.join(DATA_PATH, name))
    video_d = video_d.transpose(0, 3, 1, 2)
    audio_d = np.load(os.path.join(AUDIO_DATA_PATH, name))

    stats = {
        'min': [],
        'max': []
    }

    for j in range(video_d.shape[0]):
        stats['min'].append(video_d[j].min())
        stats['max'].append(video_d[j].max())
        video_d[j] = (video_d[j] - video_d[j].min())/(video_d[j].max() - video_d[j].min())

    video_d = torch.from_numpy(video_d)
    audio_d = torch.from_numpy(audio_d)
    print(audio_d.size())
    audio_d = audio_d.view(audio_d.size()[0], audio_d.size()[2], audio_d.size()[1])

    MINIBATCHSIZE = video_d.size()[0]

    audio_data = Variable(audio_d)
    video_data = Variable(video_d)  # this needs to be array of still frames
    still_frame = video_d[0]
    still_frame = still_frame.view(1, still_frame.size()[0], still_frame.size()[1], still_frame.size()[2])
    still_frame = Variable(still_frame.repeat(MINIBATCHSIZE, 1, 1, 1))
    noise_data = Variable(torch.rand(MINIBATCHSIZE, 1, NOISE_OUTPUT))
    noise_data = noise_data.data.resize_(noise_data.size()).normal_(0, 0.6)

    if cuda:
        #with torch.cuda.device(0):
        #unet = unet.cuda()
        #print("DEVICE 0")
        audio_data = audio_data.cuda()
        video_data = video_data.cuda()
        noise_data = noise_data.cuda()
        still_frame = still_frame.cuda()

    gen_frames = unet(still_frame, audio_data, noise_data)
    for j in range(MINIBATCHSIZE):
        if cuda:
            img = gen_frames[j].cpu().detach().numpy().transpose(1, 2, 0)
        else:
            img = gen_frames[j].detach().numpy().transpose(1, 2, 0)

        # Un normalize part will come here
        img = img*(stats['max'][j] - stats['min'][j]) + stats['min'][j]
        cv2.imwrite(SAVE_PATH+str(j)+'.png', img)

# generate_video('s9_v3_u28.npy')
# prep_video_gen_data('v1', 30, 60)

        # save_image(gen_frames, save_path+videos[i]+'.png')
