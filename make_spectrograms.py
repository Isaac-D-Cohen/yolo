import torch
import torchaudio
import torchaudio.transforms as tr
import torch.nn.functional as F

import config

import numpy as np

from sys import argv
from math import floor
import os

"""
This script takes a mode - either train or infer - and a list of file names.
The names should all correspond to .wav audio files. The files will each be chopped into
spectrograms. In inference mode the spectrograms all go in a folder called 'input'. In train mode
each file should also have a corresponding annotations file in Raven format with the same name
but the .txt extension. Then the spectrograms and labels will go in subfolders called images and labels
within a data folder.

Note: This code is largely based on the spectrograms notebook.
"""

# read a Raven annotations file into a list of lists where each inner list
# represents a box and is in the format [class, begin, end]
def read_annotations_file(annotations_filename):

    annotations = []

    with open(annotations_filename, "r") as f:
        lines_raw = [line.strip('\n').split('\t') for line in f.readlines()]

    for raw_line in lines_raw[1:]:
        time_begin = float(raw_line[3])
        time_end = float(raw_line[4])

        # we can use 0 for call and 1 for song
        obj_class = 0 if raw_line[10] == "call" else 1

        annotations.append([obj_class, time_begin, time_end])

    return annotations


def make_spectrograms(audio_name, clip_len, step):

    sound_filename = audio_name + '.wav'

    waveform, sample_rate = torchaudio.load(sound_filename)

    # waveform has shape [1, samples]
    # here we split into clips of the specified length with the specified overlap
    w = waveform.unfold(dimension=1, size=clip_len*sample_rate, step=step*sample_rate)
    # move number of clips to first dimension
    w = torch.transpose(w, 0, 1)

    # prepare our mel transform
    win_length = config.WIN_LENGTH
    hop_length = config.HOP_LENGTH
    n_fft = config.N_FFT
    n_mels = config.N_MELS

    ffts_per_sec = sample_rate/hop_length

    mel_transform = tr.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
                                    hop_length=hop_length, n_mels=n_mels)
    db_transform = tr.AmplitudeToDB()

    # convert our clips to mel spectrograms
    # the interpolation is to get the x dimension from 469 to 416
    mel_specs=[]

    for block in w:
        mel_spec = mel_transform(block)
        mel_spec = db_transform(mel_spec)
        mel_spec = F.interpolate(mel_spec, size=416, mode='linear', align_corners=False)
        mel_specs.append(mel_spec)

    return mel_specs


# insert a box into the yolo box map
# a helper function for producing the labels
def insert_box(yolo_box_map, clip_num, annotation, clip_len, step):

    # get the absolute time in the audio when a clip begins
    abs_clip_begin_time = clip_num*step
    # when does the box begin within the clip?
    box_begin_time = annotation[1]-abs_clip_begin_time
    # when does the box end?
    clip_end = abs_clip_begin_time + clip_len

    if annotation[2] <= clip_end:
        box_end_time = annotation[2]-abs_clip_begin_time
    else:
        box_end_time = clip_len     # clip_end - abs_clip_begin_time
        insert_box(yolo_box_map, clip_num+1, annotation, clip_len, step)

    # ok, now we need to normalize and find the center
    # so first normalize...
    box_begin_x = box_begin_time/clip_len
    box_end_x = box_end_time/clip_len

    # then get the center and width
    center_x = (box_begin_x+box_end_x)/2
    width = box_end_x - box_begin_x

    # and... record it!
    yolo_box = [annotation[0], center_x, width]

    if clip_num in yolo_box_map:
        yolo_box_map[clip_num].append(yolo_box)
    else:
        yolo_box_map[clip_num] = [yolo_box]

# function to produce labels
def make_labels(audio_name, clip_len, step):

    annotations_filename = audio_name + '.txt'
    annotations = read_annotations_file(annotations_filename)

    yolo_box_map = dict()    # a hashmap

    for annotation in annotations:

        # there may be multiple clips going on right now (because of overlap)
        # so first order of business is to figure out which clip started most recently
        time_begin = annotation[1]
        clip_num = floor(time_begin/step)

        while (clip_num*step + clip_len) > time_begin:
            insert_box(yolo_box_map, clip_num, annotation, clip_len, step)
            clip_num -= 1

    return yolo_box_map


def main():

    if len(argv) < 2:
        print(f"Format: {argv[0]}  <mode: either 'train' or 'infer'>  <file1>  <file2>...")
        exit(0)

    if argv[1] == "train":
        train_mode = True
    elif argv[1] == "infer":
        train_mode = False
    else:
        print(f"Error: Invalid mode {argv[1]}. Quitting...")
        exit(1)

    clip_len = config.CLIP_LEN
    overlap = config.OVERLAP
    step = clip_len-overlap

    if train_mode:
        images_dir = os.path.join("data", "images")
        labels_dir = os.path.join("data", "labels")
    else:
        images_dir = "inputs"

    for arg in argv[2:]:

        spectrograms = make_spectrograms(arg, clip_len, step)

        if train_mode:
            yolo_box_map = make_labels(arg, clip_len, step)

        for i in range(len(spectrograms)):

            spectrogram = spectrograms[i]

            # shape goes from [1, 128, 416] to [128, 416]
            spectrogram = spectrogram.view(128, 416)

            # save the tensor
            filename = os.path.join(images_dir, arg + f"_{i}.pt")
            torch.save(spectrogram, filename)

            # if we're in train mode and we have labels for this clip
            if train_mode and i in yolo_box_map:

                # save the bounding boxes
                boxes = yolo_box_map[i]

                filename = os.path.join(labels_dir, arg + f"_{i}.txt")
                with open(filename, "w") as f:
                    for box in boxes:
                        f.write(f"{box[0]} {box[1]} {box[2]}\n")


if __name__ == "__main__":
    main()
