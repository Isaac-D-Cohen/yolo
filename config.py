import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# for spectrograms
CLIP_LEN = 10   # length of a clip in seconds
OVERLAP = 5     # overlap between clips
# for fft and mel transform
WIN_LENGTH = 2048
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128

# for train/validation split
TRAIN_PORTION = 0.9

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# a minor epoch is just a run through all training data
# a major epoch is MINOR_EPOCHS number of minor epochs
# and then a validation run
MINOR_EPOCHS = 50
MAJOR_EPOCHS = 8

# loss weights (constants to weigh different parts of the loss differently)
LAMBDA_CLASS = 1
LAMBDA_NOOBJ = 16
LAMBDA_OBJ = 1
LAMBDA_BOX = 16

# for nms
PROBABILITY_THRESHOLD = .2
IOU_THRESHOLD = 0.2


CLASSES = [
           "call",
           "song"]
NUM_CLASSES=len(CLASSES)
IN_CHANNELS = 128

S = [13, 26, 52]

# Got these anchors from the k_means script
# They represent widths that are scaled to be
# proportions of the image (so between 0 and 1)
ANCHORS = [
    [0.9634, 0.8620, 0.7547],
    [0.6250, 0.4963, 0.3926],
    [0.2919, 0.1747, 0.0756]
]
