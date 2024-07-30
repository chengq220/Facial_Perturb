import torch
from kmeans_pytorch import kmeans
import numpy as np
from config import * 
import cv2
import matplotlib.pyplot as plt
from models.selection import *
import torchvision.transforms as T
import os
from models.perturbAug import AUGVGG16

def load_network(checkpoint = CHECKPOINT, checkpoint_name = CHECKPOINT_NAME):
    """
    Load network given a mdoel and its saved states
    """

    stateDict = torch.load(os.path.join(checkpoint, checkpoint_name), map_location=DEVICE)
    epoch = stateDict["epoch"]
    modelState = stateDict["model_state_dict"]
    optState = stateDict["opt_state_dict"]
    return modelState, optState, epoch

class AttentionClusterAugment(nn.Module):
    def __init__(self,alpha = 1.2, beta = 1.5, clusters = HEADS):
        super(AttentionClusterAugment, self).__init__()
        self.clusters = clusters
        self.alpha = alpha
        self.beta = beta
        self.device = DEVICE
        path = os.path.join("save/VGG16","vgg_onlygate0_moreep.tar")
        if(not os.path.isfile(path)):
            print(f"Unable to find the requested module at {path}")
            exit()
        model = AUGVGG16().to(self.device)
        modelState, _, _ = load_network("save/VGG16","vgg_onlygate0_moreep.tar")
        model.load_state_dict(modelState)
        for param in model.parameters():
            param.requires_grad = False
        self.attnMod = model.gate0
        self.cluster_img = None

    def updateClusters(self, imgs):
            if(len(imgs) != 4):
                assert("Expect 4D tensors")

            b, c, w, h = imgs.shape
            #get attention
            attn = self.attnMod(imgs).squeeze()
            attn = torch.sum(attn, dim = 0)
            attn = torch.where(attn < 0.5, 0, attn)
            attn = (attn - torch.min(attn))/(torch.max(attn)-torch.min(attn)) * self.alpha
            attn = attn.unsqueeze(2)

            #generate the 2d grid
            x_coords, y_coords = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='ij')

            #normalize x, y for numerical stability
            x_coords = x_coords.view(w,h,1).float().to(self.device)
            x_norm = (x_coords - torch.min(x_coords))/(torch.max(x_coords)-torch.min(x_coords)) * self.beta

            y_coords = y_coords.view(w,h,1).float().to(self.device)
            y_norm = (y_coords - torch.min(y_coords))/(torch.max(y_coords)-torch.min(y_coords)) * self.beta
            
            data = torch.cat((x_norm, y_norm, attn), dim=-1).reshape(-1, 3)
            #assign clusters and zero out base on clusters
            cluster_ids_x, cluster_centers = kmeans(
                X=data, num_clusters=self.clusters, distance='euclidean', device=torch.device(self.device)
            )

            cluster_centers = torch.arange(len(cluster_centers)).to(self.device)
            self.cluster_img = cluster_centers[cluster_ids_x].reshape(w, h)

                

    #Expects imgs to be in the shape of b, c, w, h
    def forward(self, imgs, update = False): 
        if(update):
            self.updateClusters(imgs)
        filters = []
        for i in range(self.clusters):
            mask = torch.where(self.cluster_img == i, False, True).float()
            alter = imgs.squeeze() * mask
            filters.append(alter)

        filters.append(imgs.squeeze())
        filters = torch.stack(filters)
        filters = filters.view(-1,filters.shape[2],filters.shape[3])
        return filters.unsqueeze(1)

if __name__ == "__main__":
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
            img = img.detach().clone().requires_grad_(True)
            label = np.zeros(NUM_CLASSES)
            label[imgInfo[0]] = 1
            label = torch.from_numpy(label).float()
            return img, label
        except FileNotFoundError as err:
            return None

    test = [[0,"data/fer2013/PrivateTest_1054527.jpg", "privatetest"],
                [1,"data/fer2013/PrivateTest_807646.jpg","privatetest"],
                [2,"data/fer2013/PrivateTest_518212.jpg","privatetest"],
                [3,"data/fer2013/PrivateTest_95094.jpg","privatetest"],
                [4,"data/fer2013/PrivateTest_59059.jpg","privatetest"],
                [5,"data/fer2013/PrivateTest_528072.jpg","privatetest"],
                [6,"data/fer2013/PrivateTest_139065.jpg","privatetest"]]
    img = []
    label = []
    for info in test:
        i, l = read_image(info)
        # print(i.shape)
        img.append(i)
        label.append(l)
    original1 = torch.stack(img) 

    center_crop = T.CenterCrop(size=(40, 40))  # Example size
    original = center_crop(original1).to(DEVICE)
    clusterMod = AttentionClusterAugment()
    out = clusterMod(original, update=True)
    print(out.shape)
    # i = 2
    # img = original[i].detach().cpu().squeeze().numpy()
    # img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_NEAREST)
    # kk = out[i][0].detach().cpu().squeeze().numpy()
    # kk = cv2.resize(kk, (100, 100), interpolation=cv2.INTER_NEAREST)
    # kk1 = out[i][1].detach().cpu().squeeze().numpy()
    # kk1 = cv2.resize(kk1, (100, 100), interpolation=cv2.INTER_NEAREST)
    # kk2 = out[i][2].detach().cpu().squeeze().numpy()
    # kk2 = cv2.resize(kk2, (100, 100), interpolation=cv2.INTER_NEAREST)

    # spacing = 10  # Adjust this value as needed

    # # Calculate canvas dimensions
    # canvas_width = 4 * 100 + 3 * spacing  # 4 images, 3 spacings
    # canvas_height = 100  # All images have the same height

    # # Create a blank canvas (white background)
    # canvas = np.ones((canvas_height, canvas_width)) * 255

    # # Place images on the canvas
    # canvas[0:100, 0:100] = img
    # canvas[0:100, 100+spacing:200+spacing] = kk
    # canvas[0:100, 200+2*spacing:300+2*spacing] = kk1
    # canvas[0:100, 300+3*spacing:400+3*spacing] = kk2

    # canvas_uint8 = (np.clip(canvas, 0, 255)*255).astype(np.uint8) 
    # plt.imshow(canvas_uint8, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')  # Turn off the axis
    # plt.savefig('masked.png', bbox_inches='tight', pad_inches=0, dpi=300)  # Save the figure