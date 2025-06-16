import torch

# takes two tensors with the lengths of lines
# returns a tensor of IOUs of the lines
def iou_length(lengths1, lengths2):

    intersection = torch.min(lengths1, lengths2)

    union = lengths1 + lengths2 - intersection

    return intersection / union
