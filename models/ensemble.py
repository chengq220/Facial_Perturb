import torch
import torch.nn as nn
from config import *
from models.selection import *

def load_network(checkpoint = CHECKPOINT, checkpoint_name = CHECKPOINT_NAME):
    """
    Load network given a mdoel and its saved states
    """

    stateDict = torch.load(os.path.join(checkpoint, checkpoint_name), map_location=DEVICE)
    epoch = stateDict["epoch"]
    modelState = stateDict["model_state_dict"]
    optState = stateDict["opt_state_dict"]
    return modelState, optState, epoch


class ensemble(nn.Module):
    def __init__(self):
        super(ensemble, self).__init__()
        self.model1 = selectModel("vgg")
        print()
        modelState1, _, _ = load_network("save/VGG16", "vgg16_6.tar")
        self.model1.load_state_dict(modelState1)

        self.model2 = selectModel("dpn")
        modelState2, _, _ = load_network("save/DPN", "dpn_1.tar")
        self.model2.load_state_dict(modelState2)

        self.model3 = selectModel("densenet")
        modelState3, _, _ = load_network("save/DenseNet", "densenet_1.tar")
        self.model3.load_state_dict(modelState3)

        self.model4 = selectModel("resnet")
        modelState4, _, _ = load_network("save/ResNet", "resnet_1.tar")
        self.model4.load_state_dict(modelState4)

        self.model5 = selectModel("resmask")
        modelState5, _, _ = load_network("save/ResMask", "resmask_1.tar")
        self.model5.load_state_dict(modelState5)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        out4 = self.model4(x)
        out5 = self.model5(x)

        return torch.sum(torch.stack([out1,out2,out3,out4,out5]), dim=0)


if __name__ == "__main__":
    model = ensemble().to(DEVICE)
    a = torch.rand((16, 1, 40, 40)).to(DEVICE)
    out = model(a)
    print(out.shape)