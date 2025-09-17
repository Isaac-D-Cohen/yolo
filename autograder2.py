import torch
import pandas as pd
from sys import argv
import os

# threshold for a box to count as present in the other list
iou_threshold = 0.75

def load_annotations_df(filename):
    if os.path.exists(filename) == False:
        print(f"Error: {filename} does not exist")
        return None
    else:
        annotations_df = pd.read_csv(filename, sep="\t")
        return annotations_df



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


# takes two tensors of shape [num_boxes, 3] where 3 is begin, end, length

# note: this function has changed since autograder 1
# TODO: remove that line if we remove autograder 1 from the repo
def compare_boxes(gt_bboxes, pred_bboxes):

    num_gt_boxes = gt_bboxes.shape[0]
    num_model_boxes = pred_bboxes.shape[0]

    if num_gt_boxes == 0 or num_model_boxes == 0:
        return 0, 0

    pred_bboxes = pred_bboxes.repeat(num_gt_boxes, 1) # (pred_bboxes, pred_bboxes, pred_bboxes...)
    gt_bboxes = gt_bboxes.unsqueeze(1).repeat(1, num_model_boxes, 1)    # ((gt_box1, gt_box1, gt_box1...), (gt_box2, gt_box2...)...)

    total_boxes = num_gt_boxes*num_model_boxes

    pred_bboxes = pred_bboxes.view(total_boxes, 3)
    gt_bboxes = gt_bboxes.view(total_boxes, 3)

    ious = intersection_over_union(gt_bboxes, pred_bboxes)

    ious = ious.view(num_gt_boxes, num_model_boxes)

    # how many of the ground truth boxes were predicted by the model
    num_gt_that_model_predicted = torch.sum(torch.max(ious, dim=1)[0] >= iou_threshold)

    ious = torch.transpose(ious, 0, 1)

    # how many of the model's output boxes correspond to a ground truth box
    num_model_boxes_that_correspond_to_gt_box = torch.sum(torch.max(ious, dim=1)[0] >= iou_threshold)

    return num_model_boxes_that_correspond_to_gt_box, num_gt_that_model_predicted


def main():

    if len(argv) < 3:
        print(f"Format: {argv[0]}\t<ground_truth_annotations>\t<model_output_annotations>")
        exit(0)

    # load the annotations
    ground_truth_df = load_annotations_df(argv[1])
    model_output_df = load_annotations_df(argv[2])

    if ground_truth_df is None or model_output_df is None:
        exit(1)

    ground_truth_df['Annotation'] = ground_truth_df['Annotation'].apply(lambda classname: classname.lower())
    model_output_df['Annotation'] = model_output_df['Annotation'].apply(lambda classname: classname.lower())

    # how many classes do we have
    classes = set(ground_truth_df['Annotation']).union(set(model_output_df['Annotation']))

    total_n1 = total_n2 = 0

    for class_name in classes:

        gt_mask = ground_truth_df['Annotation'] == class_name
        mo_mask = model_output_df['Annotation'] == class_name

        gt_class_bboxes_df = ground_truth_df.loc[gt_mask, ["Begin Time (s)", "End Time (s)", "Delta Time (s)"]]
        mo_class_bboxes_df = model_output_df.loc[mo_mask, ["Begin Time (s)", "End Time (s)", "Delta Time (s)"]]

        gt_class_bboxes = torch.tensor(gt_class_bboxes_df.to_numpy(), dtype=torch.float32)
        mo_class_bboxes = torch.tensor(mo_class_bboxes_df.to_numpy(), dtype=torch.float32)

        n1, n2 = compare_boxes(gt_class_bboxes, mo_class_bboxes)

        num_gt, num_pred = sum(gt_mask), sum(mo_mask)

        print(f"Class: {class_name}")

        if num_pred != 0:
            print(f"{n1}/{num_pred} of the model's predictions were in ground truth data, or precision = {n1/num_pred*100}%")
        else:
            print("Model predicted no boxes, so precision is undefined.")

        if num_gt != 0:
            print(f"{n2}/{num_gt} of the ground truth boxes were picked up by the model, or recall = {n2/num_gt*100}%")
        else:
            print("There were no ground truth boxes in these audio clips, so recall is undefined.")

        print()

        total_n1 += n1
        total_n2 += n2

    total_num_gt = len(ground_truth_df)
    total_num_pred = len(model_output_df)

    print("Summary:")
    print(f"{total_n1}/{total_num_pred} of the model's predictions were in ground truth data, or precision = {total_n1/total_num_pred*100}%")
    print(f"{total_n2}/{total_num_gt} of the ground truth boxes were picked up by the model, or recall = {total_n2/total_num_gt*100}%")
    print()


if __name__ == "__main__":
    main()



"""
algorithm of autograder 1:

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
