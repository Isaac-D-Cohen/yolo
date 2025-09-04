
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


def save_set_names(subset, filename):

    with open(filename, "w") as f:
        loader = DataLoader(dataset=subset, batch_size=1, shuffle=False)

        for spect in loader:
            idx = spect["idx"]
            spec_names = []
            for i in idx:
                spec_names.append(dataset.get_spect_name(i))
                f.write(f"{spec_names[0]}.pt\n")


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
            spec_names = []
            for i in idx:
                spec_names.append(dataset.get_spect_name(i))

            write_predictions(out, scaled_anchors, spec_names)
            # for m in model.modules():
            #     if isinstance(m, torch.nn.BatchNorm1d):
            #         print(m.running_mean.mean().item(), m.running_var.mean().item())

        print(f"Loss: {loss.detach()}")


if __name__ == "__main__":

    anchors = config.ANCHORS

    datadir = "data"

    images_dir = os.path.join(datadir, "images")
    labels_dir = os.path.join(datadir, "labels")

    dataset = YOLODataset(
        images_dir,
        labels_dir,
        anchors=anchors,
        S=config.S,
    )

    train_set, eval_set = random_split(dataset, [0.90, 0.1])


    save_set_names(train_set, "train_set.txt")
    save_set_names(eval_set, "eval_set.txt")

    model = YOLOv3(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)

    loss_fn = YoloLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE#, #weight_decay=config.WEIGHT_DECAY
    )

    checkpoint_name = "checkpoint16"

    # load_checkpoint(checkpoint_name, model=model)

    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm1d):
    #         print(m.running_mean.mean().item(), m.running_var.mean().item())
    #
    # exit()

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(config.S).unsqueeze(1).repeat(1, 3)
    ).to(config.DEVICE)

    for major_epoch in range(4):
        for _ in range(10):
            train_model(model, train_set, dataset, optimizer, loss_fn, scaled_anchors, training_mode=True)
        train_model(model, eval_set, dataset, optimizer, loss_fn, scaled_anchors, training_mode=False)

    train_model(model, eval_set, dataset, optimizer, loss_fn, scaled_anchors, training_mode=False, output_preds=True)

    save_checkpoint(checkpoint_name, model, optimizer)

