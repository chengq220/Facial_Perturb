import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from models.spatialAttention import SpatialGate
import torchvision.transforms as T

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU(),
        nn.Dropout(p=0.4)
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class ClusterClassifier(nn.Module):
    def __init__(self,  n_classes=7):
        super(ClusterClassifier, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.gate0 = SpatialGate()
        self.layer1 = vgg_conv_block([1,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(1*1*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):

        attn0 = self.gate0(x)
        out = x * attn0

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        vgg16_features = self.layer5(out)

        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out

if __name__ == "__main__":
    # import cv2
    # import numpy as np
    # def load_network(checkpoint = CHECKPOINT, checkpoint_name = CHECKPOINT_NAME):
    #     """
    #     Load network given a mdoel and its saved states
    #     """

    #     stateDict = torch.load(os.path.join(checkpoint, checkpoint_name), map_location=DEVICE)
    #     epoch = stateDict["epoch"]
    #     modelState = stateDict["model_state_dict"]
    #     optState = stateDict["opt_state_dict"]
    #     return modelState, optState, epoch
    

    # def read_image(imgInfo):
    #     """
    #     Read image from imgInfo 
    #     Args: 
    #         imgInfo  List containing information about the image (in the form of [class, imgPath, typeData])
    #     Returns:
    #         The image and its label
    #     """
    #     feature = cv2.imread(imgInfo[1], cv2.IMREAD_GRAYSCALE)
    #     try:
    #         if feature is None:
    #             raise FileNotFoundError("Failed to read image")
    #         img = torch.from_numpy(feature/255.0).float().unsqueeze(0)
    #         img = img.detach().clone().requires_grad_(True)
    #         label = np.zeros(NUM_CLASSES)
    #         label[imgInfo[0]] = 1
    #         label = torch.from_numpy(label).float()
    #         return img, label
    #     except FileNotFoundError as err:
    #         return None

    # b = VGG16()
    # center_crop = T.CenterCrop(size=(40, 40))  # Example size

    # # m, _, _ = load_network(checkpoint = "save/PerturbNet", checkpoint_name="perturbTry2.tar")
    # # a.load_state_dict(m)
    # test = [[0,"data/fer2013/PrivateTest_1054527.jpg", "privatetest"],
    #         [1,"data/fer2013/PrivateTest_807646.jpg","privatetest"],
    #         [2,"data/fer2013/PrivateTest_518212.jpg","privatetest"],
    #         [3,"data/fer2013/PrivateTest_95094.jpg","privatetest"],
    #         [4,"data/fer2013/PrivateTest_59059.jpg","privatetest"],
    #         [5,"data/fer2013/PrivateTest_528072.jpg","privatetest"],
    #         [6,"data/fer2013/PrivateTest_139065.jpg","privatetest"]]
    # img, l = read_image(test[1])
    # img2, l2 = read_image(test[1])
    # img = torch.stack([img, img2])
    # img = center_crop(img)
    # out, attn = b(img)
    # print(attn)
    # print(out.shape)
    # attn = a.backbone.gate0(img)
    
    # c, filters = a.peturb(img)
    # attn = a.backbone.gate0(img)
    # loss = cr(attn, c, img)
    # print(loss)
    # # print(filters.shape)
    # # c, filters = a.peturb(img)
    # filters = filters.squeeze().detach().cpu().numpy()
    # filters = filters[0,:]
    # print(filters.shape)
    # print(img.detach().cpu().squeeze().numpy().shape)
    # print(filters.shape)
    # print(filters[0].shape)
    out = np.hstack((img[0].detach().cpu().squeeze().numpy() * 255, attn.detach().cpu().squeeze().numpy() * 255))
    cv2.imwrite("manual.png", out)