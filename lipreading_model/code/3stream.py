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




def train(device, model, optimizer, datagen, epoch, epochsize, \
          s2_train_X, s3_train_X, \
          s1_train_vidlens, integral_lens):

    model.train()

    tloss=0
    for i in range(epochsize):
        X, y, vid_lens_batch, m, batch_idxs = next(datagen)
        # repeat targets based on max sequence len
        y = y.reshape((-1, 1))
        y = y.repeat(m.shape[-1], axis=-1)

        X_s2= gen_seq_batch_from_idx(s2_train_X, batch_idxs,
                                            s1_train_vidlens, integral_lens, np.max(s1_train_vidlens))

        X_s3= gen_seq_batch_from_idx(s3_train_X, batch_idxs,
                                            s1_train_vidlens, integral_lens, np.max(s1_train_vidlens))




        # print(X.shape,y.shape, vid_lens_batch.shape, m.shape,batch_idxs.shape)
    #(10, 36, 1450)     (10, 36)        (10,)               (10, 36) (10,)
        # print(X.dtype,y.dtype,vid_lens_batch.dtype,m.dtype,batch_idxs.dtype)
    # #float32          uint8        uint8                  uint8    int64

        X=torch.from_numpy(X).float().to(device)
        X_s2=torch.from_numpy(X_s2).float().to(device)
        X_s3=torch.from_numpy(X_s3).float().to(device)

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
        output,ordered_idx = model(X, X_s2, X_s3, vid_lens_batch)

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





def test(device, model,name,epoch, X_s1_val,X_s2_val, X_s3_val, y_val, \
         vid_lens_batch, mask_val, idxs_val):
    model.eval()

    y = y_val.reshape((-1, 1))
    y = y.repeat(mask_val.shape[-1], axis=-1)


    X1=torch.from_numpy(X_s1_val).float().to(device)
    X2=torch.from_numpy(X_s2_val).float().to(device)
    X3=torch.from_numpy(X_s3_val).float().to(device)

    y_val=torch.from_numpy(y_val).to(device)
    vid_lens_batch=torch.from_numpy(vid_lens_batch).to(device)
    mask_val=torch.from_numpy(mask_val).to(device)


    output,ordered_idx = model(X1, X2, X3, vid_lens_batch)


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

    ix = np.zeros((X_s1_val.shape[0],), dtype='int')
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


    parser.add_argument('--three_stream_views', type=str, default="1,2,3", help='3stream views \
    to train like 1,2,3 ; 1,2,4 ; 1,2,5 and so on. Please seperate by comma. \
    1 --> 0deg, 2 --> 30deg, 3 --> 45deg, 4 --> 60deg, 5 --> 90deg')

    parser.add_argument('--save_path', type=str, default='../results/3stream', help='path where \
    saved_model loss, acc and predictions with true lables .npy files will be saved for both test train and val ')

    parser.add_argument('--model_1stream_path', type=str, default='../results/1stream', help='path where \
    saved_model will be found for 1stream models whose weights can be used as pretrained weights for our model ')

    parser.add_argument('--data_pickle_path', type=str,  default='../data/oulu_processed.pkl', help='path where \
    data in OuluVS2 video format and processed by pre_process_oulu.py  is stored ')


    parser.add_argument('--iteration', type=int, default=1, help='if running multiple times add iteration to distinguish between \
                        different iterations ( can write a bash script and run this \
                        multiple times and save all the output to different folder by setting this to iteration no ')

    parser.add_argument('--num_epoch', type=int, default=20, help='no of epochs ')

    parser.add_argument('--num_classes', type=int, default=10, help='no of classes ')


    args = parser.parse_args()

    views =args.three_stream_views
    # views="1,2,3"
    view1=int(views.split(",")[0])
    view2=int(views.split(",")[1])
    view3=int(views.split(",")[2])

    save_path=args.save_path+"/view_"+views

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



    train_subject_ids = [1,2,3,5,7,10,11,12,14,16,17,18,19,20,21,23,24,25,27,28,31,32,33,35,36,37,39,40,41,42,45,46,47,48,53]
    val_subject_ids = [4,13,22,38,50]
    test_subject_ids = [6,8,9,15,26,30,34,43,44,49,51,52]

    model_1stream_path=args.model_1stream_path
    data_pickle_path=args.data_pickle_path


    # # on .mat file
    # imagesize1, input_dimensions1, s1_data,s1_pretrained_model= get_data(device, view1 ,shape, nonlinearities,
    #                                                  windowsize, lstm_size, model_1stream_path, args.num_classes)
    # imagesize2, input_dimensions2, s2_data,s2_pretrained_model= get_data(device, view2, shape, nonlinearities,
    #                                                  windowsize, lstm_size, model_1stream_path, args.num_classes)
    # imagesize3, input_dimensions3, s3_data,s3_pretrained_model= get_data(device, view3, shape, nonlinearities,
    #                                                  windowsize, lstm_size, model_1stream_path, args.num_classes)
    #
    # On .pkl file
    imagesize1, input_dimensions1, s1_data,s1_pretrained_model= get_data_from_file_path(device, view1 ,shape, nonlinearities,
                                                    windowsize, lstm_size, data_pickle_path, model_1stream_path, args.num_classes)
    imagesize2, input_dimensions2, s2_data,s2_pretrained_model= get_data_from_file_path(device, view2, shape, nonlinearities,
                                                    windowsize, lstm_size, data_pickle_path, model_1stream_path, args.num_classes)
    imagesize3, input_dimensions3, s3_data,s3_pretrained_model= get_data_from_file_path(device, view3, shape, nonlinearities,
                                                    windowsize, lstm_size, data_pickle_path, model_1stream_path, args.num_classes)

    print('constructing end to end model...\n')
    pretrained_stream1_model_isTrue=True

    network = adenet_3stream(device, pretrained_stream1_model_isTrue, \
                            s1_pretrained_model, shape, nonlinearities, input_dimensions1, \
                            s2_pretrained_model, shape, nonlinearities, input_dimensions2,  \
                            s3_pretrained_model, shape, nonlinearities, input_dimensions3,  \
                    windowsize, lstm_size, args.num_classes)


    network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    print("Network Architecture",network)
    print("\n")
    for key, value in network.state_dict().items():
        print(key,value.shape)
    print("\nModel on GPU",next(network.parameters()).is_cuda)


    s1_data_matrix = s1_data['dataMatrix'].astype('float32')
    s2_data_matrix = s2_data['dataMatrix'].astype('float32')
    s3_data_matrix = s3_data['dataMatrix'].astype('float32')
    targets_vec = s1_data['targetsVec'].reshape((-1,))
    subjects_vec = s1_data['subjectsVec'].reshape((-1,))
    vidlen_vec = s1_data['videoLengthVec'].reshape((-1,))

    print("Shape of data_matrix of s1:",s1_data_matrix.shape)


    matlab_target_offset=True
    meanremove= True
    samplewisenormalize= True

    #convert to 0 order
    if matlab_target_offset:
        targets_vec -= 1


    if meanremove:
        s1_data_matrix = sequencewise_mean_image_subtraction(s1_data_matrix, vidlen_vec)
        s2_data_matrix  = sequencewise_mean_image_subtraction(s2_data_matrix , vidlen_vec)
        s3_data_matrix  = sequencewise_mean_image_subtraction(s3_data_matrix , vidlen_vec)


    if samplewisenormalize:
        s1_data_matrix = normalize_input(s1_data_matrix)
        s2_data_matrix  = normalize_input(s2_data_matrix )
        s3_data_matrix  = normalize_input(s3_data_matrix )

    s1_train_X, s1_train_y, s1_train_vidlens, s1_train_subjects, \
    s1_val_X, s1_val_y, s1_val_vidlens, s1_val_subjects, \
    s1_test_X, s1_test_y, s1_test_vidlens, s1_test_subjects = split_seq_data(s1_data_matrix, targets_vec, subjects_vec,
                                                                            vidlen_vec, train_subject_ids,
                                                                            val_subject_ids, test_subject_ids)

    s2_train_X, s2_train_y, s2_train_vidlens, s2_train_subjects, \
    s2_val_X, s2_val_y, s2_val_vidlens, s2_val_subjects, \
    s2_test_X, s2_test_y, s2_test_vidlens, s2_test_subjects = split_seq_data(s2_data_matrix, targets_vec, subjects_vec,
                                                                            vidlen_vec, train_subject_ids,
                                                                            val_subject_ids, test_subject_ids)

    s3_train_X, s3_train_y, s3_train_vidlens, s3_train_subjects, \
    s3_val_X, s3_val_y, s3_val_vidlens, s3_val_subjects, \
    s3_test_X, s3_test_y, s3_test_vidlens, s3_test_subjects = split_seq_data(s3_data_matrix, targets_vec, subjects_vec,
                                                                            vidlen_vec, train_subject_ids,
                                                                            val_subject_ids, test_subject_ids)




    datagen = gen_lstm_batch_random(s1_train_X, s1_train_y, s1_train_vidlens, batchsize=batchsize)
    integral_lens = compute_integral_len(s1_train_vidlens)

    val_datagen = gen_lstm_batch_random(s1_val_X, s1_val_y, s1_val_vidlens, batchsize=len(s1_val_vidlens), shuffle=False)
    test_datagen = gen_lstm_batch_random(s1_test_X, s1_test_y, s1_test_vidlens, batchsize=len(s1_test_vidlens), shuffle=False)

    # We'll use this "validation set" to periodically check progress
    X_s1_val, y_val, vid_lens_batch_val, mask_val, idxs_val = next(val_datagen)
    integral_lens_val = compute_integral_len(s1_val_vidlens)
    X_s2_val = gen_seq_batch_from_idx(s2_val_X, idxs_val, s1_val_vidlens, integral_lens_val, np.max(s1_val_vidlens))
    X_s3_val = gen_seq_batch_from_idx(s3_val_X, idxs_val, s1_val_vidlens, integral_lens_val, np.max(s1_val_vidlens))


    # we use the test set to check final classification rate
    X_s1_test, y_test, vid_lens_batch_test, mask_test, idxs_test = next(test_datagen)
    integral_lens_test = compute_integral_len(s1_test_vidlens)
    X_s2_test = gen_seq_batch_from_idx(s2_test_X, idxs_test, s1_test_vidlens, integral_lens_test,
                                    np.max(s1_test_vidlens))
    X_s3_test = gen_seq_batch_from_idx(s3_test_X, idxs_test, s1_test_vidlens, integral_lens_test,
                                    np.max(s1_test_vidlens))


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

        #(device, model, optimizer, datagen, epoch, epochsize, s2_train_X, s3_train_X,\
    #     s1_train_vidlens, integral_lens)
        train(device, network, optimizer, datagen, epoch, epochsize, s2_train_X, s3_train_X, \
                                                s1_train_vidlens, integral_lens)


    #   train_acc, train_loss,  predictions_train, true_label_train = test(device, model,"train",epoch, X_train, y_train,  vid_lens_batch_train, mask_train, idxs_train)
        train_acc, train_loss,  predictions_train, true_label_train=(0,0,0,0)

        #(device, model,name,epoch, X_s1_val,X_s2_val, X_s3_val, y_val, vid_lens_batch, mask_val, idxs_val)
        val_acc, val_loss, predictions_val, true_label_val = test(device, model,"val",epoch, \
                                    X_s1_val,X_s2_val, X_s3_val,\
                                                                y_val, vid_lens_batch_val, mask_val, idxs_val)

        test_acc, test_loss, predictions_test, true_label_test = test(device, model,"test",epoch, \
                                X_s1_test,X_s2_test, X_s3_test,\
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
