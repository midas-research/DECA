# DECA
Data Extension and Class Addition for VSR




Supplementary files containing examples: https://drive.google.com/drive/folders/13oPBUgOG3itRcztUNI8ItyE71Zvhqe-a?usp=sharing 


# Dependencies
```
python 3.6
torch 0.4.1
CUDA 9.0
torchaudio
```

# Audio to Video Model (TC-GAN)


# VSR Model

For running the Visual Speech Recognition model, run the following command to train the model (for one view model):
```
CUDA_VISIBLE_DEVICES=1 python 1stream.py --data_pickle_path "Path to the picke file(without quotes)" --num_epoch 40 --num_classes 10  --one_stream_view 1
```
Please check the python file for details about various parameters. 

For generating the pickle file look at pre_process_oulu.py

# How to cite?

Under review

# Contact us

You can contact us at dhruva15026@iiitd.ac.in, shubham14101@iiitd.ac.in, ykumar@adobe.com
