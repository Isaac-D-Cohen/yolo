
import config
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from shutil import copyfile
import argparse

from tqdm import tqdm

from model import YOLOv3
from dataset import YOLODataset
from loss import YoloLoss
from utils import write_predictions, save_checkpoint, load_checkpoint, get_latest_checkpoint_number


def save_set_names(dataset, subset, filename):

    with open(filename, "w") as f:
        loader = DataLoader(dataset=subset, batch_size=1, shuffle=False)

        for spect in loader:
            idx = spect["idx"]
            spec_names = []
            for i in idx:
                spec_names.append(dataset.get_spect_name(i))
                f.write(f"{spec_names[0]}.pt\n")


def run_model(model, x, y, loss_fn, scaled_anchors):

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

    return out, loss


def validate_model(model, dataset, validation_set, loss_fn, scaled_anchors, output_preds=False):

    torch.autograd.set_grad_enabled(False)
    model.eval()

    # pull out all the data at once - no reason to batch for validation
    loader = DataLoader(dataset=validation_set, batch_size=len(validation_set), shuffle=False)

    val_data = next(loader)
    x, y = val_data["img"], val_data["labels"]

    out, loss = run_model(model, x, y, loss_fn, scaled_anchors)

    print(f"Loss: {loss}")

    if output_preds:
        idx = val_data["idx"]
        spec_names = []
        for i in idx:
            spec_names.append(dataset.get_spect_name(i))

        write_predictions(out, scaled_anchors, spec_names)


def train_model(model, dataset, train_set, loss_fn, optimizer, scaled_anchors, silent=False):

    torch.autograd.set_grad_enabled(True)
    model.train()

    loader = DataLoader(dataset=train_set, batch_size=config.BATCH_SIZE, shuffle=True)

    if silent:
        iterator = loader
    else:
        iterator = tqdm(loader)

    for batch in iterator:

        x, y = batch["img"], batch["labels"]

        out, loss = run_model(model, x, y, loss_fn, scaled_anchors)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if silent == False:
            print(f"Loss: {loss.detach()}")



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-with", "-s", metavar="<checkpoint_name>", help="Start with a previous checkpoint and continue training from there")
    parser.add_argument("--silent", action="store_true", help="Don't output the loss during training", default=False)
    parser.add_argument("--no-output-preds", action="store_true", help="Don't output the predictions for the validation data on the final round", default=False)
    args = parser.parse_args()

    # setup the checkpoint directory
    checkpoint_number = get_latest_checkpoint_number()+1
    checkpoint_name = f"checkpoint{checkpoint_number}"
    checkpoint_dir_path = os.path.join("checkpoints", checkpoint_name)
    os.mkdir(checkpoint_dir_path)

    # copy our config file to the checkpoint folder for future reference
    copyfile("config.py", os.path.join(checkpoint_dir_path, "config.py"))

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

    train_set, validation_set = random_split(dataset, [config.TRAIN_PORTION, 1-config.TRAIN_PORTION])


    # save a record of the train/validation split in our checkpoint folder
    save_set_names(dataset, train_set, os.path.join(checkpoint_dir_path, "train_set.txt"))
    save_set_names(dataset, validation_set, os.path.join(checkpoint_dir_path, "validation_set.txt"))

    model = YOLOv3(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)

    loss_fn = YoloLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE#, weight_decay=config.WEIGHT_DECAY
    )

    # if the wants to start with a previous checkpoint, load it
    if args.start_with != None:
        ld_checkpoint_name = args.start_with
        ld_checkpoint_path = os.path.join("checkpoints", ld_checkpoint_name)
        load_checkpoint(ld_checkpoint_path, ld_checkpoint_name, model, optimizer)

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(config.S).unsqueeze(1).repeat(1, 3)
    ).to(config.DEVICE)

    output_preds = not args.no_output_preds

    for major_epoch in range(config.MAJOR_EPOCHS):
        for _ in range(config.MINOR_EPOCHS):
            train_model(model, dataset, train_set, loss_fn, optimizer, scaled_anchors, args.silent)

        validate_model(model, dataset, validation_set, loss_fn, scaled_anchors, (major_epoch == config.MAJOR_EPOCHS-1 and output_preds))

    save_checkpoint(checkpoint_dir_path, checkpoint_name, model, optimizer)



if __name__ == "__main__":
    main()
