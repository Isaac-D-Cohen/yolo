import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# for spectrograms
# if you change these two, you should consider
# changing IMAGE_SIZE as well
CLIP_LEN = 30   # length of a clip in seconds
OVERLAP = 15     # overlap between clips
# for fft and mel transform
WIN_LENGTH = 2048
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128

REMOVE_STRADDLERS = True

# the length of each spectrogram
IMAGE_SIZE = 1248

# for dataset

# if the network uses a free anchor to predict an object
# when that object already has another anchor at this scale
# (with a better IOU), then if the IOU between the object
# and the free anchor exceeds this threshold, we won't
# punish the network for the prediction
IGNORE_IOU_THRESH = 0.5

# for train/validation split
TRAIN_PORTION = 0.9

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3

# a minor epoch is just a run through all training data
# a major epoch is MINOR_EPOCHS number of minor epochs
# and then a validation run
MINOR_EPOCHS = 50
MAJOR_EPOCHS = 8

# loss weights (constants to weigh different parts of the loss differently)
LAMBDA_CLASS = 1
LAMBDA_NOOBJ = 18
LAMBDA_OBJ = 1
LAMBDA_BOX = 4

# for nms
PROBABILITY_THRESHOLD = .2
NMS_IOU_THRESHOLD = 0.2

# for generate_annotations
NMS_WINDOW_LENGTH = 4*CLIP_LEN

# threshold for a box to count as present in the other list
AUTOGRADER_IOU_THRESHOLD = 0.5

CLASSES = [
           "call",
           "song"]

NUM_CLASSES=len(CLASSES)
IN_CHANNELS = N_MELS

S = [IMAGE_SIZE//32, IMAGE_SIZE//16, IMAGE_SIZE//8]

# Got these anchors from the k_means script
# They represent widths that are scaled to be
# proportions of the image (so between 0 and 1)

#ANCHORS = [
#    [0.5415, 0.4180, 0.3669],
#    [0.3193, 0.2568, 0.1923],
#    [0.1268, 0.0551, 0.0376]
#]

ANCHORS = [
    [0.1808, 0.1335, 0.1153],
    [0.1007, 0.0839, 0.0671],
    [0.0508, 0.0195, 0.0141]
]
