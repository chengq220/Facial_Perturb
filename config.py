#Hyperparameter
EPOCHS = 10
LR = 0.001
DECAY = 1e-4
MOMENT = 0.9
IMAGE_CHANNEL = 3
HEADS = 6
NUM_CLASSES = 7
BATCH = 16

#Training device
DEVICE = "cuda:0"

#Dataset Location
DATASET = "data/fer2013/"

#Model Selection
MODEL = "vgg"

#Saving and loading checkpoints
RESUME = False
# CHECKPOINT_NAME = "perturb_with_regloss.tar"
CHECKPOINT_NAME = "vgg16_6.tar"
CHECKPOINT = "save/VGG16/"

#Logging locations
LOG_DIR = "log/PerturbNet"
TRAIN_LOG = "perturb2024724.txt"
TEST_LOG = "perturbvgg_cluster=6.txt"

import os
if(os.path.isfile(os.path.join(CHECKPOINT,CHECKPOINT_NAME))):
    RESUME = True