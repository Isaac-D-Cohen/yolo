
import config
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from model import YOLOv3
from dataset import YOLODataset
from loss import YoloLoss


# does both training and eval
def train_model(model, loader, optimizer, loss_fn, scaled_anchors, training_mode):

    if training_mode:
        model.train()
    else:
        model.eval()


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
        else:
            print(f"Loss: {loss}")



if __name__ == "__main__":

    anchors = config.ANCHORS

    dataset = YOLODataset(
        "images/",
        "labels/",
        anchors=anchors,
        S=config.S,
    )

    train_set, eval_set = random_split(dataset, [0.95, 0.05])

    train_loader = DataLoader(dataset=train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(dataset=eval_set, batch_size=config.BATCH_SIZE, shuffle=True)

    model = YOLOv3(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)

    loss_fn = YoloLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE#, #weight_decay=config.WEIGHT_DECAY
    )

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(config.S).unsqueeze(1).repeat(1, 3)
    ).to(config.DEVICE)

    train_model(model, eval_loader, optimizer, loss_fn, scaled_anchors, training_mode=False)

    for _ in range(10):
        train_model(model, train_loader, optimizer, loss_fn, scaled_anchors, training_mode=True)
        train_model(model, eval_loader, optimizer, loss_fn, scaled_anchors, training_mode=False)

