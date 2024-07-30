from models.denseNet import *
from models.dpn import *
from models.resnet import *
from models.VGG import * 
from models.ResMasking import ResMasking50, ResMasking
# from models.perturbAug import VGG16
from config import *



def selectModel(model = MODEL):
    if(model.lower() == "vgg"):
        return VGG16().to(DEVICE)
    elif(model.lower() == "dpn"):
        return DPN26().to(DEVICE)
    elif(model.lower() == "densenet"):
        return DenseNet121().to(DEVICE)
    elif(model.lower() == "resnet"):
        return ResNet50().to(DEVICE)
    elif(model.lower() == "resmask"):
        return ResMasking().to(DEVICE)
    else:
        print("Model not recognized")
        exit(0)

if __name__ == "__main__":
    model = selectModel("ensemble")
    print(model)