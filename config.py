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

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100

# for nms
PROBABILITY_THRESHOLD = .2
IOU_THRESHOLD = 0.2


CLASSES = ["call",
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
