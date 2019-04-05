# Data files path


AUDIO_DATA_PATH = '/mnt/data/rajivratn/lipsync/data/audio'
VIDEO_DATA_PATH = '/mnt/data/rajivratn/lipsync/data/v1/train'
VIDEO_TEST_PATH = '/mnt/data/rajivratn/lipsync/data/v1/test'

# AUDIO_DATA_PATH = '/Users/vipin/codes/lipSync/data/train/audio'
# VIDEO_DATA_PATH = '/Users/vipin/codes/lipSync/data/train/video'
#
# AUDIO_TEST_PATH = '/media/data_dump_1/praveen/data_short/test/audio/'
# VIDEO_TEST_PATH = '/media/data_dump_1/praveen/data_short/test/video/'

# Audio Encoder

SEQ_LEN = 1
AUDIO_OUTPUT = 256
HIDDEN_SIZE_AUDIO = 256
NUM_LAYERS_AUDIO = 2
AUDIO_PATH = './dataset/audios/'
FRAMES_PATH = './dataset/'

# Noise Encoder

NOISE_OUTPUT = 10
BATCH = 38
HIDDEN_SIZE_NOISE = 10
NUM_LAYERS_NOISE = 1

# TRAINING PARAMS

learning_rate = 0.000001
