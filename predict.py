
import config
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

from tqdm import tqdm

from model import YOLOv3
from dataset import YOLODataset
from loss import YoloLoss
from utils import write_predictions, save_checkpoint, load_checkpoint


# does both training and eval
def train_model(model, subset, dataset, optimizer, loss_fn, scaled_anchors, training_mode, output_preds=False):

    loader = DataLoader(dataset=subset, batch_size=config.BATCH_SIZE, shuffle=True)

    if training_mode:
        torch.autograd.set_grad_enabled(True)
        model.train()
    else:
        torch.autograd.set_grad_enabled(False)
        model.eval()                     # your failing setting
        # with torch.no_grad():
        #     for m in model.modules():
        #         if isinstance(m, torch.nn.BatchNorm1d):
        #             pass
                    # m.train()            # <-- force BN back to batch mode only


    for batch in tqdm(loader):

        x = batch["img"]
        y = batch["labels"]

        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        out = model(x)
        loss = (
            loss_fn(out[0], y0, scaled_anchors[0])
            + loss_fn(out[1], y1, scaled_anchors[1])
            + loss_fn(out[2], y2, scaled_anchors[2])
        )

        if training_mode:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        elif output_preds:
            idx = batch["idx"]
            clip_nums = []
            for i in idx:
                clip_nums.append(dataset.get_spect_number(i))

            write_predictions(out, scaled_anchors, clip_nums)
            # for m in model.modules():
            #     if isinstance(m, torch.nn.BatchNorm1d):
            #         print(m.running_mean.mean().item(), m.running_var.mean().item())

        print(f"Loss: {loss.detach()}")



def main():

    anchors = config.ANCHORS

    checkpoint_filename = "checkpoints/checkpoint8.pth.tar"
    datadir = "data"

    images_dir = os.path.join(datadir, "images")
    labels_dir = os.path.join(datadir, "labels")

    dataset = YOLODataset(
        images_dir,
        labels_dir,
        anchors=anchors,
        S=config.S,
    )

    model = YOLOv3(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)

    loss_fn = YoloLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE#, #weight_decay=config.WEIGHT_DECAY
    )

    load_checkpoint(filename=checkpoint_filename, model=model, optimizer=optimizer)

    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm1d):
    #         print(m.running_mean.mean().item(), m.running_var.mean().item())
    #
    # exit()

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(config.S).unsqueeze(1).repeat(1, 3)
    ).to(config.DEVICE)


    train_model(model, dataset, dataset, optimizer, loss_fn, scaled_anchors, training_mode=False, output_preds=True)



if __name__ == "__main__":
    main()
