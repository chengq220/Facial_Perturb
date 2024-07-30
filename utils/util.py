import os 
import torch
import cv2
from config import *
import numpy as np

def save_network(model, checkpoint=CHECKPOINT, checkpoint_name=CHECKPOINT_NAME):
    """
    Save network to save folder
    """
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    save_path = os.path.join(checkpoint, checkpoint_name)
    torch.save(model, save_path)

def load_network(checkpoint = CHECKPOINT, checkpoint_name = CHECKPOINT_NAME):
    """
    Load network given a mdoel and its saved states
    """

    stateDict = torch.load(os.path.join(checkpoint, checkpoint_name), map_location=DEVICE)
    epoch = stateDict["epoch"]
    modelState = stateDict["model_state_dict"]
    optState = stateDict["opt_state_dict"]
    return modelState, optState, epoch

def read_image(imgInfo):
    """
    Read image from imgInfo 
    Args: 
        imgInfo  List containing information about the image (in the form of [class, imgPath, typeData])
    Returns:
        The image and its label
    """
    feature = cv2.imread(imgInfo[1], cv2.IMREAD_GRAYSCALE)
    try:
        if feature is None:
            raise FileNotFoundError("Failed to read image")
        img = torch.from_numpy(feature/255.0).float().unsqueeze(0)
        label = np.zeros(NUM_CLASSES)
        label[imgInfo[0]] = 1
        label = torch.from_numpy(label).float()
        return img, label
    except FileNotFoundError as err:
        return None
    
if __name__ == "__main__":
    # a,b = read_image(os.path.join(DATASET,"test/angry/PrivateTest_1054527.jpg"))
    # print(a.shape)
    # print(b.shape)
    print(0)