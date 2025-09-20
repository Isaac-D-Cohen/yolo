"""
Creates a Pytorch dataset to load our spectrograms and labels
"""

import config
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from utils import iou_length as iou, write_predictions

class YOLODataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        anchors,
        S=[13, 26, 52],
    ):

        self.img_dir = img_dir
        self.label_dir = label_dir

        self.image_filenames = os.listdir(img_dir)

        self.S = S

        # anchors comes in a list of lists, each containing 3 floats
        # so here we concatenate the lists and turn it into a tensor of shape [3, 3]
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

        self.ignore_iou_thresh = 0.5


    # The size of the dataset is the number of images
    # (or filenames of images)
    def __len__(self):
        return len(self.image_filenames)


    # This function will return an X and a Y to use during training and evaluation
    # X: a spectrogram tensor (with any audio augementations we might choose to do)
    # Y: a tuple of tensors that we can use to compute the loss function
    # During actual real life runs we will also want to know the spectrogram number
    # (a number that can tell us where in the original audio file we are). To allow
    # access to that, we will also return the index
    def __getitem__(self, index):

        # get the path
        img_path = os.path.join(self.img_dir, self.image_filenames[index])

        # load the actual spectrogram
        image = torch.load(img_path, weights_only=True)

        # normalize a little more
        image = image + (0 if image.min() >= 0 else -image.min())
        image = image/image.max()

        # see if we have labels for this image
        label_filename = self.image_filenames[index][:-2] + 'txt'
        label_path = os.path.join(self.label_dir, label_filename)

        if os.path.isfile(label_path):
            # load the labels
            # they come in as a numpy array of shape [num_labels_in_this_image, 3]
            # the 3 values are: class, x_center, width
            bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()
            return_early = False    # if we have labels we have to prepare representations of them in our y tensor
        else:
            return_early = True     # our job in this function will be much shorter


        # Now we have our spectrogram and its labels. But the labels are in the YOLO input format.
        # When we put the image through the network we're gonna get a particular type of tensor out.
        # During training we will have to compare that tensor with what we expect it to be, in order to
        # calculate the loss function.
        # The following code will translate our labels to a tuple of tensor that we can use
        # during training for this purpose. (Each tensor in the tuple will be for a different ScalePrediction layer.
        # At the end of the function, we will output our image and this set of tensors.


        # Ok, let's make a list of tensors (one for each scale) of 0s.
        # Some of these zeros will be overwritten later with other numbers; the rest will remain 0.
        # These tensors will be compared to what we get out of the network.
        # We will only flip 0 to something else if we want that position to predict an object
        targets = [torch.zeros((self.num_anchors // 3, S, 4)) for S in self.S]

        # as promised
        if return_early:
            # return an X, a Y, and an index
            return {'img': image, 'labels': tuple(targets), 'idx': index}


        # (technically... we are reserving the anchors in the order we are getting
        # to the bounding boxes. This means a particular bounding box might not get an anchor on some
        # scale because all three anchors were already taken by other objects. You can image this is more likely
        # on the 13 cell scale. The upshot is that the order in which the bounding boxes are fed in matters)
        for box in bboxes:

            # get the attributes out of this box
            # "width" is the length of the "box" (really a 1-d line)
            class_label, x, width = box

            # compute the iou with all the anchors
            # iou_anchors will have shape [9]
            iou_anchors = iou(torch.tensor(width), self.anchors)

            # anchor_indices will have the indices in descending order of iou
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)

            # we want to predict every object (detect every bird vocalization) 3 times
            # once on each scale
            # this array will keep track of whether we've already reserved
            # an anchor on a scale (one of the three scales) for this vocalization
            has_anchor = [False] * 3

            # go down the anchors in reverse order of iou
            for anchor_idx in anchor_indices:

                # since the anchors were in order of [anchors of first scale, anchors of second scale, etc.]
                # we now have an anchor index but need to figure out on which scale we are and which anchor of
                # that scale we are
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                # S will hold the scale we are in
                S = self.S[scale_idx]

                # x holds the position in the image (time in the spectrogram) of the center of the object (vocalization)
                # so S * x would tell us which cell of the S cell grid will predict this object
                j = int(S * x)

                # ok, now we can index into our giant matrix of 0s and find at [scale_idx][anchor_on_scale, j, :]
                # a set of numbers that will correspond to a prediction using this anchor

                # so first let's reach in and see if it's already being used to predict another object
                anchor_taken = targets[scale_idx][anchor_on_scale, j, 0]

                # there is a small chance that this anchor is already predicting another object
                # but if that's not happening, and we are not already using another anchor on this
                # scale to predict this object, then let's go ahead and record it
                # (whenever we assign an anchor to predict an object, we continue to the next anchor. Technically
                # this means once we find an anchor to predict our object on this scale we continue to the next anchor,
                # but unfortunately it might also be on this same scale - so we still need to check)
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, j, 0] = 1    # set the object score to 1

                    # now we need to find where the center of where our object falls within this grid cell on the image.
                    # So picture the full image and divide it into S segments. If our object is in segment 6 and 0.25
                    # way through, then S*x will give us 6.25, and we can subtract j (6) to get 0.25
                    x_cell = S * x - j

                    # find the width
                    # in our labels this is expressed as a proportion of the entire image
                    # but for model output we will get an absolute pixel number relate to our current grid size.
                    # So if we have 13 cells in our output tensor, we will get the width in terms of those
                    width_cell = width * S      # this can be greater than 1 since it's relative to cell

                    # stick in our results
                    targets[scale_idx][anchor_on_scale, j, 1] = x_cell
                    targets[scale_idx][anchor_on_scale, j, 2] = width_cell

                    # stick in the class label
                    targets[scale_idx][anchor_on_scale, j, 3] = int(class_label)

                    # mark down that we used an anchor on this scale to predict the image! yay!
                    has_anchor[scale_idx] = True

                # if this anchor box is free, but we don't need it because we already have an anchor at this scale
                # then if the iou of this anchor box with our bounding box is greater than some threshold
                # we don't punish the network for predicting this bounding box using this anchor box
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, j, 0] = -1  # ignore prediction

        # return an X, a Y, and an index
        return {'img': image, 'labels': tuple(targets), 'idx': index}


    # Calling this function will give you the name of
    # a spectrogram file (without the .pt) for a given index
    def get_spect_name(self, index):
        img_filename = self.image_filenames[index]
        # chop off the .pt
        return img_filename[:-3]


def test():

    S = config.S
    anchors = config.ANCHORS

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(config.S).unsqueeze(1).repeat(1, 3)
    ).to(config.DEVICE)

    dataset = YOLODataset(
        "images/",
        "labels/",
        anchors=config.ANCHORS,
        S=S,
    )

    loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    for batch in loader:

        x = batch["img"]
        y = batch["labels"]
        idx = batch["idx"]

        spec_names = []
        for i in idx:
            spec_names.append(dataset.get_spect_name(i))

        # print(x.shape)
        # print(len(y))
        # print(y[0].shape)
        # print(f"Idx = {idx}")

        write_predictions(list(y), scaled_anchors, spec_names, is_preds=False)


if __name__ == "__main__":
    test()
