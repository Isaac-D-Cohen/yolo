import torch
import config

# takes two tensors with the lengths of lines
# returns a tensor of IOUs of the lines
def iou_length(lengths1, lengths2):

    intersection = torch.min(lengths1, lengths2)

    union = lengths1 + lengths2 - intersection

    # this is instead of adding a constant during division
    # ne_0 is short for "not equal to 0"
    ne_0 = union != 0

    return_tensor = torch.zeros(union.shape)
    return_tensor[ne_0] = intersection[ne_0] / union[ne_0]

    return return_tensor

# takes two tensors of lines, each tensor having shape [batch_size, 2]
# where the second dimension holds an x value and a
# width value for each line. The x value is the midpoint.
# It returns a tensor of IOUs of the lines
def intersection_over_union(lines1, lines2):

    lines1_begin = lines1[:,0:1] - lines1[:,1:2]/2
    lines1_end = lines1[:,0:1] + lines1[:,1:2]/2
    lines2_begin = lines2[:,0:1] - lines2[:,1:2]/2
    lines2_end = lines2[:,0:1] + lines2[:,1:2]/2

    begin = torch.max(lines1_begin, lines2_begin)
    end = torch.min(lines1_end, lines2_end)

    intersection = (end-begin).clamp(0)
    union = lines1[:,1:2] + lines2[:,1:2] - intersection

    # this is instead of adding a constant during division
    # ne_0 is short for "not equal to 0"
    ne_0 = union != 0

    return_tensor = torch.zeros(union.shape)
    return_tensor[ne_0] = intersection[ne_0] / union[ne_0]

    return return_tensor


# This function is almost exactly a verbatim copy of Aladdin Persson's version.
# If you want a good explanation of the subject and the code, definitely check out his video:
# https://youtu.be/YDkjWEN8jNA

# bboxes is a list of bounding boxes for a particular image
# if a box's confidence is below probability_threshold we exclude it from the start
# then, we exclude all boxes that have the same class as some box with a greater probability
# and their IOU with that box is greater than iou_threshold
def non_max_suppression(bboxes):

    probability_threshold = config.PROBABILITY_THRESHOLD
    iou_threshold = config.IOU_THRESHOLD

    bboxes = [box for box in bboxes if box[1] > probability_threshold]
    bboxes.sort(key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]).unsqueeze(0),
                torch.tensor(box[2:]).unsqueeze(0),
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def format_output(bboxes, clip_num):

    final_boxes = []

    for box in bboxes:
        class_label, confidence_x_iou, x_center, width = box

        x_begin = x_center - width/2
        x_begin, width = x_begin*10+clip_num*5, width*10
        x_end = x_begin+width

        final_boxes.append([class_label, confidence_x_iou, x_begin, x_end, width])

    final_boxes.sort(key=lambda box: box[2])

    return final_boxes


# predictions is a list of tensors each holding the outputs for an entire batch for one scale
# shape of each item in the list is [batch_size, anchors_per_scale, S, 3+C]
# anchors is a tensor containing the anchors for each scale. shape is [num_scales, anchors_per_scale]
# clip_nums is a list of the id numbers of the spectrograms in this batch
# (you can get this info from the dataset with dataset.get_spect_number())
def write_predictions(predictions, scaled_anchors, clip_nums, is_preds=True):

    batch_size = predictions[0].shape[0]
    anchors_per_scale = scaled_anchors.shape[1]

    # we need as many lists for results as we have batch_size
    bboxes = [[] for _ in range(batch_size)]

    # scale_predictions is a tensor of shape [batch_size, anchors_per_scale, S, 3+C)]
    for scale_ind, scale_predictions in enumerate(predictions):

        S = scale_predictions.shape[2]

        if is_preds:
            # reshape the anchors to match the number of dimensions in our scale_predictions tensor
            # [batch_size, anchors_per_scale, S, 3+C]
            current_scale_anchors = scaled_anchors[scale_ind].view(1, 3, 1, 1)

            # sigmoid the x center and confidence predictions
            scale_predictions[..., 0:2] = torch.sigmoid(scale_predictions[..., 0:2])
            # scale the width predictions
            scale_predictions[..., 2:3] = (
                torch.exp(scale_predictions[..., 2:3]) * current_scale_anchors
                )

            # get the class prediction from the logits
            class_predictions = torch.argmax(scale_predictions[..., 3:], dim=-1)

        else:
            class_predictions = scale_predictions[..., 3]

        # loop over the cells at this scale
        for scale_idx in range(S):
            # scale the x center predictions
            # this is the opposite of x_cell = S * x - j (see dataset.py)
            scale_predictions[..., scale_idx, 1] = (scale_predictions[..., scale_idx, 1] + scale_idx)/S

            # scale the width predictions
            scale_predictions[..., scale_idx, 2] = scale_predictions[..., scale_idx, 2]/S

            # loop over the batch
            for i in range(batch_size):
                # and the anchors
                for anchor_idx in range(anchors_per_scale):

                    # if the object IOU score is not 0, we want to record this prediction
                    if scale_predictions[i, anchor_idx, scale_idx, 0] != 0:

                        confidence_x_iou = scale_predictions[i, anchor_idx, scale_idx, 0].item()
                        x_center = scale_predictions[i, anchor_idx, scale_idx, 1].item()
                        width = scale_predictions[i, anchor_idx, scale_idx, 2].item()
                        class_label = class_predictions[i, anchor_idx, scale_idx].item()

                        # record in the list for this clip (that's what bboxes[i] indexes)
                        bboxes[i].append([class_label, confidence_x_iou, x_center, width])

    # ok, now it's time to remove all the extra bounding boxes with non maximal suppression
    # and record the remaining ones in txt files
    # remember that bboxes has shape [batch_size, detections_in_this_clip, 4]
    for i in range(batch_size):
        # apple non maximal suppression and format for output
        final_boxes = format_output(non_max_suppression(bboxes[i]), clip_nums[i])

        # and now we record our results
        with open(f"outputs/{clip_nums[i]}.txt", "w") as f:
            for box in final_boxes:
                line = " ".join([str(item) for item in box])
                f.write(line)
                f.write('\n')


# TODO Maybe modify these functions a bit
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr
