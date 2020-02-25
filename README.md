# DECA
Data Extension and Class Addition for VSR

Code for paper: **Harnessing GANs for Addition of New Classes in VSR**

Link to paper: https://drive.google.com/open?id=13xofvtwqOAXWgjOYu8EFaeMQBTKiB0T7

Link to the supplementary folder (Appendix pdf, paper pdf ,and example reconstructed video with embedded audio) : https://drive.google.com/open?id=1nMgRuyM9juztd_fOzTH62X1v1eTh4R5u

Folder containing all generated examples: https://drive.google.com/drive/folders/13oPBUgOG3itRcztUNI8ItyE71Zvhqe-a?usp=sharing 


# Dependencies
```
python 3.6
torch 0.4.1
CUDA 9.0
torchaudio
```

# Audio to Video Model (TC-GAN)

For running the A2V model, run the following command
```
CUDA_VISIBLE_DEVICES=1 python driver.py
``` 
For training and video generation change following lines:

* Epochs-> line 78 | num_epochs (variable name)

* View-> line 257 | videos, audios = get_data("v1", 1) --> change v1 to v2,v3,v4,v5 to train for different views, and value 1 can go upto the number of speakers you want to train for (30 or 60) in our case .


* comment out line 277 -> For forward pass on the model for video generation
* uncomment line 420 -> generate_video('s9_v3_u28.npy') --> namme of the folder containing reference image also the name of the folder containing reference audio file to generate video
* uncomment line 421 -> prep_video_gen_data('v1', 30, 60) --> change v1 to v2,v3,v4,v5 to generate videos for different views and set range of speakers anywhere between 0-60 in our case to generate videos specifically for those speakers for that particular view


# VSR Model

For running the Visual Speech Recognition model, run the following command to train the model (for one view model):
```
CUDA_VISIBLE_DEVICES=1 python 1stream.py --data_pickle_path "Path to the picke file(without quotes)" \
--num_epoch 40 \
--num_classes 10  \
--one_stream_view 1
```
Please check the python file for details about various parameters. 

For generating the pickle file look at pre_process_oulu.py

# How to cite?

@misc{kumar2019harnessing,
    title={Harnessing GANs for Zero-shot Learning of New Classes in Visual Speech Recognition},
    author={Yaman Kumar and Dhruva Sahrawat and Shubham Maheshwari and Debanjan Mahata and Amanda Stent and Yifang Yin and Rajiv Ratn Shah and Roger Zimmermann},
    year={2019},
    eprint={1901.10139},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

# Contact us

You can contact us at dhruva15026@iiitd.ac.in, shubham14101@iiitd.ac.in, ykumar@adobe.com
