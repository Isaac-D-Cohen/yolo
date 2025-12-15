# About this project

There are many classifiers for bird vocalizations. If you have a recording and want to know what species it is, you can use a classifier such as BirdNet or HawkEars. However, if you want to do detection - that is you have a clip and want to know where in the clip a bird sings - your options are currently limited. I'm only aware of two papers so far that created systems for detection of bird vocalizations. You can find them [here](https://reference-global.com/article/10.2478/orhu-2019-0015) and [here](https://www.sciencedirect.com/science/article/pii/S1574954125002638). Their code repos are [here](https://github.com/zsebok/YOLO) and [here](https://github.com/GrunCrow/BIRDeep_BirdSongDetector_NeuralNetworks), respectively. The present repo provides another such detection system.

In computer vision, YOLO (You Only Look Once) is an architecture that can detect (find) objects in photos. This project adapts YOLO to work on audio files and find bird calls and songs in them. We primarily tested with Northern Cardinal audio from Upstate New York.

Note: I wrote this program to help animals. Please do not use it in projects (conservation or otherwise) that hurt wild animal welfare. For those unfamiliar with the concept, see [here](https://www.wildanimalinitiative.org/faq). (Disclaimer: I am not affiliated with, nor do I speak for the Wild Animal Initiative. But they're a great organization and you should totally get involved with their work.)

# How to run the code

## Input/Output

For training: To train a model you need a set of audio files in WAV format and a corresponding set of Raven annotations files.

For inference: To run the model you need an audio files (or multiple audio files). The program will generate an output file in Raven format.

## Get Raven

To annotate audio or view the output annotations, you'll need Raven (Lite/Pro). You can download Raven Lite [here](https://www.ravensoundsoftware.com/software/raven-lite/).

## Setup the repo

Download or clone this repo and install the requirements using `pip install -r requirements.txt` or the equivalent command on your favorite python package manager.

## To perform training

1) Create two folders, one called `audio` and the other called `annotations`. In the `audio` folder, create a folder and place your `.wav` files inside. In `annotations`, create a corresponding folder with the same name and place your `.txt` annotations there. (To get the annotations from Raven, you need to click `Export Selection Table`.)

2) Run `python make_spectrograms.py train <name of your folder in audio and annotations>`. The make_spectrograms scripts will create a `data` folder with `images` and `labels` folders inside, containing the pytorch tensors and their labels in YOLO format all ready for training.

3) Optional but recommended: Run `python k_means.py` and copy the resulting anchors into `config.py` replacing the current ones. This will customize the anchors for your dataset.

4) Run `python train.py`. This script will train a new model. It also has some options you can see with `python train.py --help`. I recommend you run this script on a computer with a good GPU. Once it is done, you will have a `checkpoints` folder and in that a `checkpoint<n>` folder, where `n` starts at 1 and goes up each time you store a new model. The newly created `outputs` folder will have predictions for the validation data that were made at the end of training.

5) Optional: If you want to see the predictions for the validation data, run `python generate_annotations.py` to turn them into a Raven annotations file.

## To perform inference

I will describe here the steps to use the command line, but there's also a GUI you can use in `predict.pyw`.

For the command line:

1) Create a folder called `audio` and within it a folder where you place your `.wav` files.

2) Run `python make_spectrograms.py infer <name of your folder in audio>`. This will create a folder called `inputs` with pytorch tensors ready to be fed into the model.

3) Run `python predict.py` optionally giving it a checkpoint name as an argument. The checkpoint name would be `checkpoint<n>` where `n` is the number associated with your checkpoint. These are assigned sequentially as described in the previous section.

4) The outputs are now in the folder `outputs`. Run `python generate_annotations.py` to get them in Raven format.

# Credits and contact

Professor Justin Mann gave me annotated audio of Northern Cardinals (recorded in Upstate NY) to train and test this on. The idea to use YOLO came from Professor Jayson Boubin. Professor Ken Chiu was my advisor for this project. The YOLO code is an adaptation of Aladdin Persson's YOLOv3 implementation, and I am very much indebted to him for his video which helped me understand and implement YOLO. You can watch it [here](https://youtu.be/Grir6TZbc1M?si=OfYivJUzfwI2PBiE).

Throughout this code the comments may use regular YOLO terminology, like referring to "bounding boxes." Technically the tensors here have shape (mel bins, time axis), with no Y (the mel bins are the channels). Still I call them "boxes" for simplicity.

You can contact me at: idcohen2000 [at] gmail [dot] com.