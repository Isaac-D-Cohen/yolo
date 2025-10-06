import torch
import torchaudio
import torchaudio.transforms as tr
import torch.nn.functional as F

import config

import pandas as pd

from sys import argv
from math import floor
import os

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

# read a Raven annotations file and return a list of lists where each inner list
# represents a box and is in the format [class_index, begin, end]
def read_annotations_file(annotations_filename):

    annotations_df = pd.read_csv(annotations_filename, sep='\t')
    important_columns = annotations_df.loc[:, ["Annotation", "Begin Time (s)", "End Time (s)"]]

    # map our class strings to integer indices using the classes array from the config file
    # (we convert to lowercase first because sometimes annotators enter "Song" or "SONG" instead of "song")
    important_columns["Annotation"] = important_columns["Annotation"].apply(lambda classname: config.CLASSES.index(classname.lower()))

    return important_columns.values.tolist()


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
        silence_tensor = torch.zeros((1, samples_short))
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

    mel_transform = tr.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
                                    hop_length=hop_length, n_mels=n_mels)
    db_transform = tr.AmplitudeToDB()

    # convert our clips to mel spectrograms
    # the interpolation is to get the x dimension from 469 to 416
    mel_specs=[]

    for block in w:
        mel_spec = mel_transform(block)
        mel_spec = db_transform(mel_spec)
        mel_spec = F.interpolate(mel_spec, size=config.IMAGE_SIZE, mode='linear', align_corners=False)
        mel_specs.append(mel_spec)

    return mel_specs


# insert a box into the yolo box map
# a helper function for producing the labels
def insert_box(yolo_box_map, clip_num, annotation, clip_len, step):

    # get the absolute time in the audio when a clip begins
    abs_clip_begin_time = clip_num*step
    # when does the clip end?
    abs_clip_end_time = abs_clip_begin_time + clip_len


    # when does the box begin within the clip?
    # (don't worry if the annotation extends into a previous clip;
    # we already took care of that outside this function call)
    box_begin_time = max(0, annotation[1]-abs_clip_begin_time)

    # when does the box end within the clip? (clip_len = clip_end - abs_clip_begin_time)
    box_end_time = min(annotation[2]-abs_clip_begin_time, clip_len)

    # ok, now we need to normalize and find the center
    # so first normalize...
    box_begin_x = box_begin_time/clip_len
    box_end_x = box_end_time/clip_len

    # then get the center and width
    center_x = (box_begin_x+box_end_x)/2
    width = box_end_x - box_begin_x

    # One last thing before we record
    # In order to:
    # 1) prevent duplicate boxes in the same clip
    # 2) allow us to later erase boxes (with their underlying
    # parts of the clip) that straddle a clip boundary
    # we want to record the beginning and end within this clip
    # and the true length of the annotation box
    true_len = annotation[2] - annotation[1]

    # ok, let's package together all the info
    current_box = [annotation[0], center_x, width, box_begin_time, box_end_time, true_len]

    if clip_num in yolo_box_map:
        # since there are other boxes here already, we must search for duplicates
        for box in yolo_box_map[clip_num]:
            # if we found an identical box, defined as a box that starts and ends in the
            # same place within this clip and has the same class, just return
            if box[3] == current_box[3] and box[4] == current_box[4] and box[0] == current_box[0]:
                return
        # if we got here, record it!
        yolo_box_map[clip_num].append(current_box)
    else:
        # we can just put it in
        yolo_box_map[clip_num] = [current_box]


# function to produce labels
def make_labels(annotations_filename, clip_len, step):

    annotations = read_annotations_file(annotations_filename)

    yolo_box_map = dict()    # a hashmap

    for annotation in annotations:

        # there may be multiple clips going on right now (because of overlap)
        # so first order of business is to figure out which clip started most recently
        # (before the end of our annotation)
        _, time_begin, time_end = annotation
        clip_num = floor(time_end/step)

        # in each clip that ends after our this annotation starts
        while (clip_num*step + clip_len) > time_begin:
            # we want to insert our annotation
            insert_box(yolo_box_map, clip_num, annotation, clip_len, step)
            clip_num -= 1

    return yolo_box_map

# remove a box and the boxes that overlap with it
# note: removing a box means clearing its associated audio
# also note: these two functions are written like this because the list
# may change significantly when we stop on an if statement and go down the
# recursion rabbit hole
def remove_box(spectrogram, boxes, box_to_remove):

    _, _, _, box_begin_time, box_end_time, _ = box_to_remove

    boxes.remove(box_to_remove)

    # blank out area on spectrogram
    box_begin_sample = int(box_begin_time*config.IMAGE_SIZE/config.CLIP_LEN)
    box_end_sample = int(box_end_time*config.IMAGE_SIZE/config.CLIP_LEN)

    # just in case
    box_begin_sample = max(box_begin_sample, 0)
    box_end_sample = min(box_end_sample, config.IMAGE_SIZE-1)

    # zero it out
    spectrogram[:, box_begin_sample:box_end_sample+1] = 0

    # now we need to deal with the other boxes that overlap with this one
    while True:
        for other_box in boxes:
            _, _, _, other_box_begin_time, other_box_end_time, _ = other_box

            # if the other box overlaps with the one we just erased
            if box_begin_time < other_box_end_time and box_end_time > other_box_begin_time:
                remove_box(spectrogram, boxes, other_box)
                break
        else:
            return

# remove all boxes from this spectrogram that straddle its boundaries
def remove_straddlers(spectrogram, boxes):

    while True:
        for box in boxes:
            _, _, _, box_begin_time, box_end_time, true_len = box

            # if the box's true length is longer than its time in this clip
            if true_len > (box_end_time - box_begin_time):
                remove_box(spectrogram, boxes, box)
                break
        else:
            return



def main():

    if len(argv) < 3:
        print(f"Format: {argv[0]}  <mode: either 'train' or 'infer'>  <directory with files>")
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
            spectrogram = spectrogram.view(config.N_MELS, config.IMAGE_SIZE)

            # if we're in train mode and we have labels for this clip
            if train_mode and i in yolo_box_map:

                # save the bounding boxes
                boxes = yolo_box_map[i]

                remove_straddlers(spectrogram, boxes)

                # after removing the straddlers and their overlappers we may not have any annotations left
                if len(boxes) > 0:
                    filename = os.path.join(labels_dest, sound_name + f"_{i}.txt")
                    with open(filename, "w") as f:
                        for box in boxes:
                            f.write(f"{int(box[0])} {box[1]} {box[2]}\n")

            # save the tensor
            filename = os.path.join(images_dest, sound_name + f"_{i}.pt")
            torch.save(spectrogram, filename)


if __name__ == "__main__":
    main()
