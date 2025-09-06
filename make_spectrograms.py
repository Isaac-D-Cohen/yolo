import torch
import torchaudio
import torchaudio.transforms as tr
import torch.nn.functional as F

import config

import numpy as np

from sys import argv
from math import floor
import os

from config import CLASSES as classes

"""
This script takes a mode - either train or infer - and a directory name. A directory with this
name should exist in audio/ and maybe annotations/ too. The one in audio/ should contain .wav files;
the one in annotations/ - if it exists - should have .txt files with corresponding names, each having
the Raven annotations for the audio file with its name. The clips will each be chopped into spectrograms.
In inference mode the spectrograms all go in a folder called 'input' and we don't need the annotations.
In train mode, the script will additionally create YOLO style labels from the annotation files and then put
the spectrograms in data/images and the labels in data/labels.

Note: This code is largely based on the spectrograms notebook.
"""

def ensure_dirs_exist(tuple_of_dir_names):

    for dir_name in tuple_of_dir_names:
        if not os.path.isdir(dir_name):
            print(f"Error: {dir_name} doesn't exist.")
            exit(1)

# read a Raven annotations file into a list of lists where each inner list
# represents a box and is in the format [class, begin, end]
def read_annotations_file(annotations_filename):

    annotations = []

    with open(annotations_filename, "r") as f:
        lines_raw = [line.strip('\n').split('\t') for line in f.readlines()]

    for raw_line in lines_raw[1:]:
        time_begin = float(raw_line[3])
        time_end = float(raw_line[4])

        # map our class string to an integer using the classes array from the config file
        obj_class = classes.index(raw_line[10])

        annotations.append([obj_class, time_begin, time_end])

    return annotations


def make_spectrograms(audio_filename, clip_len, step):

    waveform, sample_rate = torchaudio.load(audio_filename)

    # if this sound file is stereo, drop one channel
    if waveform.shape[0] == 2:
        waveform = waveform[0:1,:]

    samples_per_clip = clip_len*sample_rate

    # is this sound file under the length of one clip (10 seconds)?
    # how many samples short are we?
    samples_short = samples_per_clip - waveform.shape[1]

    # if we are under 10 seconds, add some silence as filler
    if samples_short > 0:
        silence_tensor = torch.zeroes(samples_short)
        waveform = torch.cat((waveform, silence_tensor), dim=1)


    # waveform has shape [1, samples]
    # here we split into clips of the specified length with the specified overlap
    w = waveform.unfold(dimension=1, size=samples_per_clip, step=step*sample_rate)
    # move number of clips to first dimension
    w = torch.transpose(w, 0, 1)

    # we probably have some residual samples that didn't make it into any window
    num_clips = w.shape[0]
    # clip_len seconds for the first clip, and then 'step' seconds for the remaining ones
    end_of_last_clip = (clip_len + (num_clips-1)*step)*sample_rate
    residual = waveform.shape[1] - end_of_last_clip

    if residual > 0:
        new_last_clip = waveform[0,-samples_per_clip:]
        new_last_clip = new_last_clip.view(1, 1, -1)
        w = torch.cat((w, new_last_clip), dim=0)


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
def make_labels(annotations_filename, clip_len, step):

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

    if len(argv) < 3:
        print(f"Format: {argv[0]}  <mode: either 'train' or 'infer'>  <directory with files>...")
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

    images_src = os.path.join("audio", argv[2])
    ensure_dirs_exist((images_src,))

    if train_mode:
        labels_src = os.path.join("annotations", argv[2])
        images_dest = os.path.join("data", "images")
        labels_dest = os.path.join("data", "labels")
        ensure_dirs_exist((labels_src, images_dest, labels_dest))
    else:
        images_dest = "inputs"
        ensure_dirs_exist((images_dest,))

    wav_files = os.listdir(images_src)

    for wav_basename in wav_files:

        wav_name = os.path.join(images_src, wav_basename)
        spectrograms = make_spectrograms(wav_name, clip_len, step)

        # remove the .wav
        sound_name = wav_basename[:-4]

        if train_mode:
            annotations_filename = os.path.join(labels_src, sound_name + '.txt')
            yolo_box_map = make_labels(annotations_filename, clip_len, step)

        for i in range(len(spectrograms)):

            spectrogram = spectrograms[i]

            # shape goes from [1, 128, 416] to [128, 416]
            spectrogram = spectrogram.view(128, 416)

            # save the tensor
            filename = os.path.join(images_dest, sound_name + f"_{i}.pt")
            torch.save(spectrogram, filename)

            # if we're in train mode and we have labels for this clip
            if train_mode and i in yolo_box_map:

                # save the bounding boxes
                boxes = yolo_box_map[i]

                filename = os.path.join(labels_dest, sound_name + f"_{i}.txt")
                with open(filename, "w") as f:
                    for box in boxes:
                        f.write(f"{box[0]} {box[1]} {box[2]}\n")


if __name__ == "__main__":
    main()
