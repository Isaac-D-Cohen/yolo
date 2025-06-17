import torch

# takes two tensors with the lengths of lines
# returns a tensor of IOUs of the lines
def iou_length(lengths1, lengths2):

    intersection = torch.min(lengths1, lengths2)

    union = lengths1 + lengths2 - intersection

    # this is instead of adding a constant during division
    # e_0 is short for "equal to 0"
    e_0 = union == 0

    return_tensor = torch.empty(union.shape)
    return_tensor[e_0] = torch.tensor(0)
    return_tensor[~e_0] = intersection[~e_0] / union[~e_0]

    return return_tensor

# takes two tensors of lines, each tensor having shape [batch_size, 2]
# where the second dimension holds an x value and a
# width value for each line. The x value is the midpoint.
# It returns a tensor of IOUs of the lengths of the lines
def intersection_over_union(lines1, lines2):

    lines1_begin = lines1[:,0:1] - lines1[:,1:2]/2
    lines1_end = lines1[:,0:1] + lines1[:,1:2]/2
    lines2_begin = lines2[:,0:1] - lines2[:,1:2]/2
    lines2_end = lines2[:,0:1] + lines2[:,1:2]/2

    begin = torch.max(lines1_begin, lines2_begin)
    end = torch.min(lines1_end, lines2_end)

    intersection = (begin-end).clamp(0)
    union = lines1[:,1:2] + lines2[:,1:2] - intersection

    # this is instead of adding a constant during division
    # e_0 is short for "equal to 0"
    e_0 = union == 0

    return_tensor = torch.empty(union.shape)
    return_tensor[e_0] = torch.tensor(0)
    return_tensor[~e_0] = intersection[~e_0] / union[~e_0]

    return return_tensor

