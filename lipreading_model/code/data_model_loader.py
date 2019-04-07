import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import glob
from models import *
from collections import OrderedDict

def pretrained_cutoff_deltanet_majority_vote(device, loaded_model_dict, shapes, nonlinearities, input_size,window,
                                             hidden_units,output_classes=10):
    custom_model=cutoff_deltanet_majority_vote(device, False, None, shapes, nonlinearities,
                                               input_size,window, hidden_units, output_classes)
    # assumed that loaded_model_dict is on cpu

    new_state_dict= OrderedDict()
    for key, value in custom_model.state_dict().items():
        new_state_dict[key]=loaded_model_dict[key]

    custom_model.load_state_dict(new_state_dict)




    return custom_model




def get_best_model(device, fpath, shapes, nonlinearities, input_size,window,
                                             hidden_units,output_classes=10):
    files=glob.glob(fpath+"/*")
    max_acc=0
    best_model_dict=None
    for i in files:
        loaded_dict=torch.load(i,
                           map_location=lambda storage, loc: storage)
#         print(loaded_dict["test_accuracy"])
        if max_acc < loaded_dict["val_accuracy"]:
            max_acc= loaded_dict["val_accuracy"]
            best_model_dict=loaded_dict["state_dict"]

    best_model=pretrained_cutoff_deltanet_majority_vote(device, best_model_dict, shapes, nonlinearities, input_size,window,
                                             hidden_units,output_classes)
    return best_model,max_acc

def get_data(device, view, shapes, nonlinearities,window,
                                             hidden_units, model_1stream_path="../results/1stream",output_classes=10):
    if view==1:
        imagesize= [29,50]
        input_size= 1450
        data=sio.loadmat("../data/allMouthROIsResized_frontal.mat")
        pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_1/models", shapes, nonlinearities, input_size,window,
                                             hidden_units,output_classes)

    elif view==2:
        imagesize= [29,44]
        input_size= 1276
        data=sio.loadmat("../data/allMouthROIsResized_30.mat")
        pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_2/models", shapes, nonlinearities, input_size,window,
                                             hidden_units,output_classes)

    elif view==3:
        imagesize= [29,43]
        input_size= 1247
        data=sio.loadmat("../data/allMouthROIsResized_45.mat")
        pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_3/models", shapes, nonlinearities, input_size,window,
                                             hidden_units,output_classes)

    elif view==4:
        imagesize= [35,44]
        input_size= 1540
        data=sio.loadmat("../data/allMouthROIsResized_60.mat")
        pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_4/models", shapes, nonlinearities, input_size,window,
                                             hidden_units,output_classes)

    elif view==5:
        imagesize= [44,30]
        input_size= 1320
        data=sio.loadmat("../data/allMouthROIsResized_profile.mat")
        pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_5/models", shapes, nonlinearities, input_size,window,
                                             hidden_units,output_classes)

    print("Loaded view "+str(view)+" model with val acc: ",acc)
    return imagesize, input_size, data, pretrained_1stream_model


def get_data_from_file_path(device, view, shapes, nonlinearities,window,
                                                            hidden_units,data_file_path="../data/oulu_processed.pkl",
                                                                model_1stream_path="../results/1stream", output_classes=10):

    with open(data_file_path, "rb") as myFile:
        data_processed= pickle.load(myFile)

        if view==1:
            imagesize= [29,50]
            input_size= 1450
            data = data_processed[view]
            pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_1_iteration_*/models", \
                                                shapes, nonlinearities, input_size,window, \
                                             hidden_units,output_classes)
        elif view==2:
            imagesize= [29,44]
            input_size= 1276
            data = data_processed[view]
            pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_2_iteration_*/models", \
                                                shapes, nonlinearities, input_size,window, \
                                             hidden_units,output_classes)
        elif view==3:
            imagesize= [29,43]
            input_size= 1247
            data = data_processed[view]
            pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_3_iteration_*/models", \
                                                shapes, nonlinearities, input_size,window, \
                                             hidden_units,output_classes)
        elif view==4:
            imagesize= [35,44]
            input_size= 1540
            data = data_processed[view]
            pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_4_iteration_*/models", \
                                                shapes, nonlinearities, input_size,window, \
                                             hidden_units,output_classes)
        elif view==5:
            imagesize= [44,30]
            input_size= 1320
            data = data_processed[view]
            pretrained_1stream_model,acc=get_best_model(device, model_1stream_path+"/view_5_iteration_*/models", \
                                                shapes, nonlinearities, input_size,window, \
                                             hidden_units,output_classes)

    print("Loaded view "+str(view)+" model with acc: ",acc)
    return imagesize, input_size, data, pretrained_1stream_model
