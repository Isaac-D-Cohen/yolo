
import torch
import torch.nn as nn

from utils import intersection_over_union
from config import LAMBDA_CLASS, LAMBDA_NOOBJ, LAMBDA_OBJ, LAMBDA_BOX

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants to weigh different parts of the loss differently
        self.lambda_class = LAMBDA_CLASS
        self.lambda_noobj = LAMBDA_NOOBJ
        self.lambda_obj = LAMBDA_OBJ
        self.lambda_box = LAMBDA_BOX


    # target is one of the tensors in the list we output from the dataset class
    # (specifically, it's the one corresponding to this scale - the loss is called for
    # each scale). predictions is the output from the model for this scale.
    # it has shape [batch, anchors_per_scale, cells_in_this_scale (S), num_classes+3]
    def forward(self, predictions, target, anchors):
        # We make two tensors of true/false values marking which anchors
        # are supposed to predict an object (in this target tensor) and which aren't.
        # If target == -1 we don't list it in either tensor (it's false in both)
        # because we don't want to make the network use it, but we also don't want to "punish"
        # the network for using it
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # loss for when there is no object, but the network predicted one
        no_object_loss = self.bce(
            (predictions[..., 0][noobj]), (target[..., 0][noobj]),
        )

        # now, there's a special case where there might not be any objects at all
        # if so, we already covered all bases and can return the loss now
        if not torch.any(obj):
            return self.lambda_noobj * no_object_loss

        # ok, in the next two sections we're going to need the x center predictions
        # so let's sigmoid them now. The reason we sigmoid in the first place is to keep them in range [0, 1]
        predictions[..., 1] = self.sigmoid(predictions[..., 1])

        # loss for when there is an object but the network predicted the wrong iou
        # that is, it put the box somewhere and gave an iou for its box and the real one
        # but, the iou of its box and the actual one is not equal to its predicted iou
        # so either change your coordinates or change your iou estimate!
        # (in the next section we will specifically correct the box position and size, so ideally the iou should tend towards 1)

        # so first we reshape the anchors to match the number of dimensions in our predictions tensor
        # new shape = [batch, num_anchors, cells_in_this_scale, prediction_values]
        anchors = anchors.reshape(1, 3, 1, 1)


        # then we concatenate the predictions for x_center with the predictions for width to get
        # predictions for the lines. The exponent stuff is just how the YOLO paper said to do things
        line_preds = torch.cat([predictions[..., 1:2], torch.exp(predictions[..., 2:3]) * anchors], dim=-1)

        # now let's get the IOUs between our line predictions and the lines we have in target
        # we detatch just in case having the gradients flow through the IOUs messes things up
        ious = intersection_over_union(line_preds[obj], target[..., 1:3][obj]).detach()

        # finally, we compute the loss for this part by comparing our IOU predicitons to the actual IOUs
        # as derived from our predicted lines and the ground truth lines
        # this error is how far off our IOUs predictions were given our x and width predictions and the ground truth
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])


        # loss for when there is an object and the network predicted it
        # but put it in the wrong place

        # to improve gradient flow stuff, we transform the targets to match the predictions format
        # instead of the other way around. So this log thing of the width in target is the inverse
        # of the exponent thing on the width predictions above
        target[..., 2:3] = torch.log(
            (1e-16 + target[..., 2:3] / anchors)
        )

        # compute the loss for this part by comparing our predictions for x and width to targets
        box_loss = self.mse(predictions[..., 1:3][obj], target[..., 1:3][obj])

        # finally, loss for when there is an object and the network predicted it, but it got the class wrong
        class_loss = self.entropy(
            (predictions[..., 3:][obj]), (target[..., 3][obj].long()),
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
