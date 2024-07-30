import torchvision.transforms as T
import torch
from models.selection import selectModel
from utils.util import load_network
from config import *
import cv2

def predict(imagePath):
    model = selectModel()
    modelState, _, _ = load_network()
    model.load_state_dict(modelState)
    model.eval()
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    transform = T.Compose([
        T.ToTensor(),
        T.CenterCrop((40,40)),
        T.Normalize(mean=(0,), std=(255,))
    ])
    ccm = transform(image).unsqueeze(0).to(DEVICE)
    pred = model(ccm)
    return torch.argmax(pred)


if __name__ == "__main__":
    pred = predict("data/fer2013/Training_3309884.jpg") #surprise
    # pred = predict("data/fer2013/PrivateTest_59059.jpg") #neutral
    # pred = predict("data/fer2013/PrivateTest_95094.jpg") #happy

    # pred = predict("data/fer2013/PrivateTest_518212.jpg") #fear
    # pred = predict("data/fer2013/PrivateTest_528072.jpg") #sad
    # pred = predict("data/fer2013/PrivateTest_1054527.jpg") #angry
    # pred = predict("data/fer2013/PrivateTest_807646.jpg") #disgust

    print(pred)