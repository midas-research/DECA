import cv2
import os
import glob
import shutil
import numpy as np
import pickle
import scipy.io as sio
from collections import OrderedDict
import pickle
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from models import *
from data_model_loader import *
from utils import *
from loss_datagen import *




def train(device, model, optimizer, datagen, epoch, epochsize):

    model.train()

    tloss=0
    for i in range(epochsize):
        X, y, vid_lens_batch, m, batch_idxs = next(datagen)
        # repeat targets based on max sequence len
        y = y.reshape((-1, 1))
        y = y.repeat(m.shape[-1], axis=-1)


        # print(X.shape,y.shape, vid_lens_batch.shape, m.shape,batch_idxs.shape)
    #(10, 36, 1450)     (10, 36)        (10,)               (10, 36) (10,)
        # print(X.dtype,y.dtype,vid_lens_batch.dtype,m.dtype,batch_idxs.dtype)
    # #float32          uint8        uint8                  uint8    int64

        X=torch.from_numpy(X).float().to(device)
        y=torch.from_numpy(y).long().to(device)
        vid_lens_batch=torch.from_numpy(vid_lens_batch).to(device)
        m=torch.from_numpy(m).float().to(device)

#         X.to(device)
#         y.to(device)
#         vid_lens_batch.to(device)
#         m.to(device)
        # batch_idxs.to(device)

        optimizer.zero_grad()

#         print(X.is_cuda)
#         print(next(model.parameters()).is_cuda)
#         print(X.shape,X_s2.shape)
        output,ordered_idx = model(X, vid_lens_batch)

        target=torch.index_select(y,0,ordered_idx)
        m=torch.index_select(m,0,ordered_idx)

        loss = temporal_ce_loss(output, target,m)


        tloss+=loss.item()

        loss.backward()

        #gradient clip, if needed add
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, tloss*1.0/epochsize))





def test(device, model,name,epoch, X_val, y_val, \
         vid_lens_batch, mask_val, idxs_val):
    model.eval()

    y = y_val.reshape((-1, 1))
    y = y.repeat(mask_val.shape[-1], axis=-1)


    X=torch.from_numpy(X_val).float().to(device)
    y_val=torch.from_numpy(y_val).to(device)
    vid_lens_batch=torch.from_numpy(vid_lens_batch).to(device)
    mask_val=torch.from_numpy(mask_val).to(device)


    output,ordered_idx = model(X, vid_lens_batch)


    y_val=torch.index_select(y_val,0,ordered_idx)
    mask_val=torch.index_select(mask_val,0,ordered_idx)


    y=torch.from_numpy(y).long().to(device)
    target=torch.index_select(y,0,ordered_idx)


    m=mask_val.float()

    loss = temporal_ce_loss(output, target,m)

    output=output.cpu().detach().numpy()

    seq_len=output.shape[1]
    y_val=y_val[:].contiguous()
    mask_val=mask_val[:,:seq_len].contiguous()

    mask_val=mask_val.cpu().numpy()
    y_val=y_val.cpu().numpy()

    num_classes = output.shape[-1]

    ix = np.zeros((X_val.shape[0],), dtype='int')
    seq_lens = np.sum(mask_val, axis=-1)



    # for each example, we only consider argmax of the seq len
    votes = np.zeros((num_classes,), dtype='int')
    for i, eg in enumerate(output):
        predictions = np.argmax(eg[:seq_lens[i]], axis=-1)
#         print(predictions.shape)
        for cls in range(num_classes):
            count = (predictions == cls).sum(axis=-1)
            votes[cls] = count
        ix[i] = np.argmax(votes)


    c = ix == y_val
#     print(c,ix[:10],y_val[:10])
    classification_rate = np.sum(c == True) / float(len(c))


    print('{} Epoch: {} \tAcc: {:.6f} \tLoss: {:.6f}'.format(name,
                    epoch,classification_rate,loss.item() ))

    preds = ix
    true_labels = y_val

    return classification_rate,loss.item(), preds, true_labels





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default="0", help='Please write which gpu to use 0 is for cuda:0 and so one \
                        if you want to use CPU specify -1 ')

    parser.add_argument('--one_stream_view', type=str, default="1", help='1stream view \
    to train like 1, 2 and so on .   \
    1 --> 0deg, 2 --> 30deg, 3 --> 45deg, 4 --> 60deg, 5 --> 90deg')

    parser.add_argument('--save_path', type=str, default='../results/1stream', help='path where \
    saved_model loss, acc and predictions with true lables .npy files will be saved for both test train and val ')

    parser.add_argument('--pretrained_encoder_path', type=str, default='../pretrained_encoder/', help='path where pretrained encoder \
    is found ')

    parser.add_argument('--data_pickle_path', type=str,  default='../data/oulu_processed.pkl', help='path where \
    data in OuluVS2 video format and processed by pre_process_oulu.py  is stored ')


    parser.add_argument('--iteration', type=int, default=1, help='if running multiple times add iteration to distinguish between \
                        different iterations ( can write a bash script and run this \
                        multiple times and save all the output to different folder by setting this to iteration no ')

    parser.add_argument('--num_epoch', type=int, default=20, help='no of epochs ')
    parser.add_argument('--num_classes', type=int, default=10, help='no of classes ')



    args = parser.parse_args()

    view =args.one_stream_view
    view=int(view)

    save_path=args.save_path+"/view_"+str(view)

    if args.iteration:
        save_path+="_iteration_"+str(args.iteration)

    os.makedirs(save_path,exist_ok=True)
    os.makedirs(save_path+"/models",exist_ok=True)
    os.makedirs(save_path+"/predictions_truelabels",exist_ok=True)



# for setting gpu and Cpu
#
# either this
# gpu_no=0
# torch.cuda.set_device(gpu_no)
#and use .cuda()
#
# or this
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# and replacing every .cuda() with .to(device)

    gpu=args.gpu
    device_name='cpu'

    if gpu>=0:
        device_name='cuda:'+str(gpu)
    device = torch.device(device_name)


    shape= [2000,1000,500,50]
    nonlinearities= ["rectify","rectify","rectify","linear"]
    # preprocessing options
    reorderdata= False
    diffimage= False
    meanremove= True
    samplewisenormalize= True
    featurewisenormalize= False

    # [lstm_classifier]
    windowsize= 3
    lstm_size= 450
    output_classes= args.num_classes

    matlab_target_offset= True

    #[training]
    learning_rate= 0.0003
#     num_epoch= 40
    num_epoch=args.num_epoch
    epochsize= 105*2 
    batchsize= 10

    #35*30=1050
    #35*39=1365

    print("no of epochs:",num_epoch)

    train_subject_ids = [1,2,3,5,7,10,11,12,14,16,17,18,19,20,21,23,24,25,27,28,31,32,33,35,36,37,39,40,41,42,45,46,47,48,53]
    val_subject_ids = [4,13,22,38,50]
    test_subject_ids = [6,8,9,15,26,30,34,43,44,49,51,52]

    pretrained_encoder_path=args.pretrained_encoder_path
    data_pickle_path=args.data_pickle_path

    with open(data_pickle_path, "rb") as myFile:
        data_processed= pickle.load(myFile)


    if view==1:
        imagesize= [29,50]
        input_dimensions= 1450
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_frontal.mat"

    elif view==2:
        imagesize= [29,44]
        input_dimensions= 1276
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_30.mat"

    elif view==3:
        imagesize= [29,43]
        input_dimensions= 1247
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_45.mat"

    elif view==4:
        imagesize= [35,44]
        input_dimensions= 1540
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_60.mat"

    elif view==5:
        imagesize= [44,30]
        input_dimensions= 1320
        data = data_processed[view]
        stream1=pretrained_encoder_path + "/oulu_relu_ae_mean_removed20ep_trainData_profile.mat"



    print('constructing end to end model...\n')
    pretrained_encoder_isTrue=True

    #ae
    pre_trained_encoder_variables = load_decoder(stream1, shape)

    network=deltanet_majority_vote(device, pretrained_encoder_isTrue, \
                pre_trained_encoder_variables, shape, nonlinearities, input_dimensions, windowsize, lstm_size, args.num_classes)


    network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    print("Network Architecture",network)
    print("\n")
    for key, value in network.state_dict().items():
        print(key,value.shape)
    print("\nModel on GPU",next(network.parameters()).is_cuda)


    data_matrix = data['dataMatrix'].astype('float32')
    targets_vec = data['targetsVec'].reshape((-1,))
    subjects_vec = data['subjectsVec'].reshape((-1,))
    vidlen_vec = data['videoLengthVec'].reshape((-1,))

    print("Shape of data_matrix of s1:",data_matrix.shape)


    matlab_target_offset=True
    meanremove= True
    samplewisenormalize= True

    #convert to 0 order
    train_X, train_y, train_vidlens, train_subjects, \
    val_X, val_y, val_vidlens, val_subjects, \
    test_X, test_y, test_vidlens, test_subjects =split_seq_data(data_matrix, targets_vec, subjects_vec, vidlen_vec,
                                                                    train_subject_ids, val_subject_ids, test_subject_ids)

    if matlab_target_offset:
        train_y -= 1
        val_y -= 1
        test_y -= 1

    if meanremove:
        train_X = sequencewise_mean_image_subtraction(train_X, train_vidlens)
        val_X = sequencewise_mean_image_subtraction(val_X, val_vidlens)
        test_X = sequencewise_mean_image_subtraction(test_X, test_vidlens)

    if samplewisenormalize:
        train_X = normalize_input(train_X)
        val_X = normalize_input(val_X)
        test_X = normalize_input(test_X)




    datagen = gen_lstm_batch_random(train_X, train_y, train_vidlens, batchsize=batchsize)

    val_datagen = gen_lstm_batch_random(val_X, val_y, val_vidlens, batchsize=len(val_vidlens), shuffle=False)
    test_datagen = gen_lstm_batch_random(test_X, test_y, test_vidlens, batchsize=len(test_vidlens), shuffle=False)
    train_datagen= gen_lstm_batch_random(train_X, train_y, train_vidlens, batchsize=len(train_vidlens), shuffle=False)


    # We'll use this "validation set" to periodically check progress
    X_val, y_val,  vid_lens_batch_val, mask_val, idxs_val = next(val_datagen)

    # we use the test set to check final classification rate
    X_test, y_test,  vid_lens_batch_test, mask_test, idxs_test = next(test_datagen)

    #get train accuracy
    #     X_train, y_train,  vid_lens_batch_train, mask_train, idxs_train = next(train_datagen)


    # train_datagen= gen_lstm_batch_random(s1_train_X, s1_train_y, s1_train_vidlens, batchsize=len(s1_train_vidlens), shuffle=False)
    # X_train, y_train,  vid_lens_batch_train, mask_train, idxs_train = next(train_datagen)



    loss_list_train=[]
    loss_list_test=[]
    loss_list_val=[]

    acc_list_train=[]
    acc_list_test=[]
    acc_list_val=[]
    model=network

    tstart = time.time()

    print("Started training")

    for epoch in range(1,num_epoch+1):
        time_start = time.time()

        #(device, model, optimizer, datagen, epoch, epochsize)
        train(device, network, optimizer, datagen, epoch, epochsize)


    #   train_acc, train_loss,  predictions_train, true_label_train = test(device, model,"train",epoch, X_train, y_train,  vid_lens_batch_train, mask_train, idxs_train)
        train_acc, train_loss,  predictions_train, true_label_train=(0,0,0,0)

        #(device, model,name,epoch, X_val, y_val, vid_lens_batch, mask_val, idxs_val)
        val_acc, val_loss, predictions_val, true_label_val = test(device, model,"val",epoch, \
                                    X_val,\
                                                                y_val, vid_lens_batch_val, mask_val, idxs_val)

        test_acc, test_loss, predictions_test, true_label_test = test(device, model,"test",epoch, \
                                X_test,\
                                                                y_test, vid_lens_batch_test, mask_test, idxs_test)

        print("Time Taken for epoch:",epoch," ",time.time()-time_start)

        if (epoch%5==0 ):
            print("Saved model dict at epoch:",epoch," at path ",save_path)
            state={
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),

                'test_loss':             test_loss,
                'test_accuracy':         test_acc,
                'test_predictions':      predictions_test,
                'test_true_label':       true_label_test,

                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_predictions':      predictions_train,
                'train_true_label':       true_label_train,

                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_predictions':      predictions_val,
                'val_true_label':       true_label_val,

#                 'optimizer' : optimizer.state_dict(),
            }
            fname=save_path+"/models/epoch_"+str(epoch)+"_checkpoint.pth.tar"
            torch.save(state,fname)

        # np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_prediction_train.npy", predictions_train)
        np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_prediction_test.npy", predictions_test)
        np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_prediction_val.npy", predictions_val)
        # np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_true_label_train.npy", true_label_train)
        np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_true_label_test.npy", true_label_test)
        np.save(save_path+"/predictions_truelabels/epoch_"+str(epoch)+"_true_label_val.npy", true_label_val)

        # loss_list_train.append(train_loss)
        loss_list_test.append(test_loss)
        loss_list_val.append(val_loss)

        # acc_list_train.append(train_acc)
        acc_list_test.append(test_acc)
        acc_list_val.append(val_acc)

        # np.save(save_path+"/loss_list_train.npy", loss_list_train)
        np.save(save_path+"/loss_list_test.npy", loss_list_test)
        np.save(save_path+"/loss_list_val.npy", loss_list_val)


        # np.save(save_path+"/acc_list_train.npy", acc_list_train)
        np.save(save_path+"/acc_list_test.npy", acc_list_test)
        np.save(save_path+"/acc_list_val.npy", acc_list_val)





    print(time.time()-tstart)








if __name__ == "__main__":
    main()
