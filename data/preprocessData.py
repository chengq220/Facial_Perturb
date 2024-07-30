import os
import numpy as np
import pandas as pd
import shutil
import json
from config import *

def preprocessData():
    label = []
    imgPath = []
    classes = []

    if(not os.path.isdir(DATASET)):
        os.makedirs(DATASET)

    with open(os.path.join(DATASET, "classes.txt"), 'r') as file:
        classDict = file.read()

    try:
        classDict = json.loads(classDict)
    except:
        print("failed to read in dictionary")

    roots = []
    image_extensions=('jpg', 'jpeg', 'png', 'gif')
    for root, directories, files in os.walk(DATASET):
        if any(file.lower().endswith(image_extensions) for file in files):
            roots.append(root)

    for rootPath in roots:
        currLabel = os.path.basename(rootPath)
        for _, _, files in os.walk(rootPath):
            for file in files:
                fileName, ext = os.path.splitext(file)
                if(ext == ".jpg"):
                    classes.append((fileName.split("_")[0]).lower())
                    label.append(classDict[currLabel])
                    shutil.copy(os.path.join(rootPath, file), os.path.join(DATASET, file))
                    imgPath.append(os.path.join(DATASET, file))

    df = pd.DataFrame({
            'label': label,
            'path': imgPath,
            'classes': classes
    }) 

    df.to_csv(DATASET + 'dataset.csv', index=False)
    return 0


if __name__ == "__main__":
    preprocessData()
