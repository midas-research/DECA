import pickle
import cv2
import numpy as np
import glob
import pprint
import os
import imresize
import argparse


def get_target(num):

    if num in [31,32,33]:
        trg = 1
    elif num in [34,35,36]:
        trg = 2
    elif num in [37,38,39]:
        trg = 3
    elif num in [40,41,42]:
        trg = 4
    elif num in [43,44,45]:
        trg = 5
    elif num in [46,47,48]:
        trg = 6
    elif num in [49,50,51]:
        trg = 7
    elif num in [52,53,54]:
        trg = 8
    elif num in [55,56,57]:
        trg = 9
    elif num in [58,59,60]:
        trg = 10
    return trg

parser = argparse.ArgumentParser()

parser.add_argument('--oulu_path', type=str, default="../data/cropped_mouth_mp4_phrase/", help=' path to oulu dataset')

parser.add_argument('--save_path', type=str, default='../data/oulu_processed.pkl', help=' path with filename for where and with what name you want to store the processed Oulu dataset')

args = parser.parse_args()


data_path=args.oulu_path
save_path=args.save_path

a={}

print("Starting data processing...\n")
for i in range(1,6):
    print("View:",i)
    a[i]={}
    
    if i==1:
        targetH = 29
        targetW = 50
    elif i==2:
        targetH = 29
        targetW = 44
    elif i==3:
        targetH = 29
        targetW = 43
    elif i==4:
        targetH = 35
        targetW = 44
    elif i==5:
        targetH = 44
        targetW = 30
        
    datamatrix_cell=[]
    targetsVec=[]
    subjectsVec=[]
    videoLengthVec=[]
    filename=[]
        
    for j in range(1,54):
        
        if j==29:
            continue
        videos=glob.glob(data_path+str(j)+"/"+str(i)+"/*")
        videos.sort()
        tcount=0
        for k in videos:
            
            filename.append(k.split("/")[-1])
            subjectsVec.append(int(k.split("/")[-1].split("_")[0][1:]))
            
            vidcap = cv2.VideoCapture(k)
            success,image = vidcap.read()
            count = 0
            gray_mat=[]
            while success:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_gray=img_gray.astype(float)
                img_gray_resize=imresize.imresize(img_gray, output_shape=(targetH, targetW))
                gray_mat.append(img_gray_resize.flatten('F'))
                targetsVec.append(get_target(int(k.split("/")[-1].split("_")[2].split(".")[0][1:])))
                
                
                success,image = vidcap.read()
                count += 1
                
            datamatrix_cell.append(np.array(gray_mat))
            videoLengthVec.append(count)
            tcount+=count
            
        

    data_matrix=np.vstack(datamatrix_cell)
    a[i]["filename"]=filename
    a[i]["dataMatrix"]=data_matrix
    a[i]["targetsVec"]=np.array(targetsVec)
    a[i]["subjectsVec"]=np.array(subjectsVec )
    a[i]["videoLengthVec"]=np.array(videoLengthVec)
    
# for testing
# for i in range(1,6):
#     print(len(a[i]["filename"]),a[i]["filename"][100:110])
#     print(a[i]["dataMatrix"].shape)
#     print(a[i]["targetsVec"].shape,a[i]["targetsVec"][0:65])
#     print(a[i]["subjectsVec"].shape,a[i]["subjectsVec"][0:31])
#     print(a[i]["videoLengthVec"].shape,a[i]["videoLengthVec"][0:31])


with open(save_path, "wb") as myFile:
    pickle.dump(a, myFile)

    