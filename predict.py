import config
import torch
from torch.utils.data import Dataset, DataLoader
import os

from tqdm import tqdm

from model import YOLOv3
from utils import write_predictions, load_checkpoint

# a dataset class just to just load spectrograms
class InputDataset(Dataset):
    def __init__(self, input_dir="inputs"):

        self.input_dir = input_dir
        self.image_filenames = os.listdir(input_dir)

    def __getitem__(self, index):
        # get the path
        img_path = os.path.join(self.input_dir, self.image_filenames[index])
        # load the spectrogram
        image = torch.load(img_path, weights_only=True)
        # normalize
        image = (image - image.mean())/image.std()
        # return the spectrogram, and its index
        return {'img': image, 'idx': index}

    def __len__(self):
        return len(self.image_filenames)

    # Calling this function will give you the name of
    # a spectrogram file (without the .pt) for a given index
    def get_spect_name(self, index):
        img_filename = self.image_filenames[index]
        # chop off the .pt
        return img_filename[:-3]



def clear_outputs():

    output_dir = "outputs"

    for filename in os.listdir(output_dir):
        p = os.path.join(output_dir, filename)
        os.remove(p)

def main():

    clear_outputs()

    torch.autograd.set_grad_enabled(False)

    anchors = config.ANCHORS

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(config.S).unsqueeze(1).repeat(1, 3)
    ).to(config.DEVICE)

    checkpoint_name = "checkpoint20"

    dataset = InputDataset()
    loader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE)
    model = YOLOv3(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)

    load_checkpoint(checkpoint_name, model=model)
    model.eval()


    for spectrogram_batch in tqdm(loader):

        spectrograms = spectrogram_batch["img"]
        spectrograms = spectrograms.to(config.DEVICE)

        preds = model(spectrograms)

        idx = spectrogram_batch["idx"]
        spec_names = []
        for i in idx:
            spec_names.append(dataset.get_spect_name(i))

        write_predictions(preds, scaled_anchors, spec_names)


if __name__ == "__main__":
    main()
