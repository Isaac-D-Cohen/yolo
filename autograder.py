import torch
import numpy as np
from sys import argv
import warnings

# threshold for a box to count as present in the other list
iou_threshold = 0.7

# threshold for a box to count as having been predicted by the model
# (we will eliminate all boxes below this threshold for our analysis)
confidence_threshold = 0.2

def read_ground_truth_bboxes(filename):

    with open(filename, "r") as f:
        lines_raw = [line.strip('\n').split('\t') for line in f.readlines()]

    lines = [
            [0.0 if line[9] == "call" else 1.0, float(line[3]), float(line[4]), float(line[7])]
            for line in lines_raw[1:]
        ]

    return torch.tensor(lines)


# takes two tensors of lines, each tensor having shape [batch_size, 3]
# where the second dimension is begin, end, length.
# It returns a tensor of IOUs of the lines
def intersection_over_union(lines1, lines2):

    begin = torch.max(lines1[:, 0:1], lines2[:, 0:1])
    end = torch.min(lines1[:, 1:2], lines2[:, 1:2])

    intersection = (end-begin).clamp(0)
    union = lines1[:,2:3] + lines2[:,2:3] - intersection

    # this is instead of adding a constant during division
    # ne_0 is short for "not equal to 0"
    ne_0 = union != 0

    return_tensor = torch.zeros(union.shape)
    return_tensor[ne_0] = intersection[ne_0] / union[ne_0]

    return return_tensor


def compare_boxes(gt_bboxes, pred_bboxes):

    num_gt_boxes = gt_bboxes.shape[0]
    num_model_boxes = pred_bboxes.shape[0]

    if num_gt_boxes == 0 or num_model_boxes == 0:
        return 0, 0

    pred_bboxes = pred_bboxes.repeat(num_gt_boxes, 1) # (pred_bboxes, pred_bboxes, pred_bboxes...)
    gt_bboxes = gt_bboxes.unsqueeze(1).repeat(1, num_model_boxes, 1)    # ((gt_box1, gt_box1, gt_box1...), (gt_box2, gt_box2...)...)

    total_boxes = num_gt_boxes*num_model_boxes

    pred_bboxes = pred_bboxes.view(total_boxes, 5)
    gt_bboxes = gt_bboxes.view(total_boxes, 4)

    ious = intersection_over_union(gt_bboxes[:,1:], pred_bboxes[:, 2:])

    ious = ious.view(num_gt_boxes, num_model_boxes)

    # how many of the ground truth boxes were predicted by the model
    num_gt_that_model_predicted = torch.sum(torch.max(ious, dim=1)[0] >= iou_threshold)

    ious = torch.transpose(ious, 0, 1)

    # how many of the model's output boxes correspond to a ground truth box
    num_model_boxes_that_correspond_to_gt_box = torch.sum(torch.max(ious, dim=1)[0] >= iou_threshold)

    return num_model_boxes_that_correspond_to_gt_box, num_gt_that_model_predicted


def process_spectrogram(gt_bboxes, num):

    spectrogram_number = int(num)

    t_start = spectrogram_number*5
    t_end = t_start + 10

    # find the ground truth boxes that are in this spectrogram
    starts_after_begin_mask = gt_bboxes[:,1] >= t_start
    ends_before_end_mask = gt_bboxes[:,2] <= t_end
    mask = torch.logical_and(starts_after_begin_mask, ends_before_end_mask)
    masked_gt_boxes = gt_bboxes[mask]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_bboxes = torch.from_numpy(np.loadtxt('outputs/' + num + '.txt', delimiter=" ", ndmin=2, dtype=np.float32))

    if len(pred_bboxes) == 0:
        return 0, 0, pred_bboxes.shape[0], masked_gt_boxes.shape[0]

    pred_bboxes = pred_bboxes[pred_bboxes[:,1] >= confidence_threshold]    # filter out the boxes below confidence_threshold

    n1, n2 = compare_boxes(masked_gt_boxes, pred_bboxes)

    return n1, n2, pred_bboxes.shape[0], masked_gt_boxes.shape[0]


def main():

    total_n1 = total_n2 = total_num_pred = total_num_gt = 0

    # returns a tensor of shape [num_boxes, 4] where 4 is class, start, end, length
    gt_bboxes = read_ground_truth_bboxes("dr_manns_annotations.txt")

    for arg in argv[1:]:

        n1, n2, num_pred, num_gt = process_spectrogram(gt_bboxes, arg)

        print(f"Spectrogram: {arg}")
        print(f"{n1}/{num_pred} of the model's predictions were in ground truth data")
        print(f"{n2}/{num_gt} of the ground truth boxes were picked up by the model")
        print()

        total_n1 += n1
        total_n2 += n2
        total_num_pred += num_pred
        total_num_gt += num_gt

    print("Summary:")
    print(f"{total_n1}/{total_num_pred} of the model's predictions were in ground truth data")
    print(f"{total_n2}/{total_num_gt} of the ground truth boxes were picked up by the model")
    print()


if __name__ == "__main__":
    main()



"""
algorithm:

- read the mann annotations file into a tensor of shape [lines, attributes]
- filter for only lines within the current spectrogram
- read in the results from the model
- cutoff model boxes with too low confidence
- repeat the model boxes so each one occurs num_gt_boxes times
- reshape to be [num_model_boxes, num_gt_boxes, attributes]
- repeat gt boxes model_boxes times
- reshape to be [num_gt_boxes, num_model_boxes, attributes]
- get ious for these two tensors. resulting tensor is 1-d with len [num_model_boxes*num_gt_boxes]
- reshape resulting tensor to [num_model_boxes, num_gt_boxes]
- see how many we got and return the answers
"""
