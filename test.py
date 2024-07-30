import torch
from dataset import *
import numpy as np
import matplotlib.pyplot as plt
from config import *
from utils.util import load_network
from models.selection import *
from utils.metrics import *
import os
from tqdm import tqdm
from models.ensemble import ensemble
from models.clustering import AttentionClusterAugment


def test():
    model = selectModel()
    modelState, _, _ = load_network()
    model.load_state_dict(modelState)
    # model = ensemble()
    _, _, testLoader = getDataloaders()

    model.eval()
    batchRep = np.zeros(NUM_CLASSES, dtype=np.float64)
    accuracy = []
    f1 = []

    clusterMod = AttentionClusterAugment()
    update = True
    with torch.no_grad():
        for i, batch in tqdm(enumerate(testLoader), total=len(testLoader)):
            img_batch, label_batch = batch
            img_batch = img_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            bs, ncrops, c, h, w = img_batch.shape
            img_batch = img_batch.view(-1, c, h, w)
            # label_batch = torch.repeat_interleave(label_batch, repeats=ncrops, dim=0)

            pred = model(img_batch, clusterMod, update=update)
            b, c = pred.shape
            pred = pred.reshape(HEADS+1, -1, c).permute(1,0,2)
            pred= torch.sum(pred, dim = 1)
            pred = pred.view(bs, ncrops, -1)
            pred = torch.sum(pred, dim=1) / ncrops

            classAcc, rep = classAccuracy(pred, label_batch)
            batchRep[rep] += 1
            accuracy.append(classAcc)
            classF1 = f1ScorePerClass(pred, label_batch)
            f1.append(classF1)
            # update = False

    accuracy = np.array(accuracy)
    f1 = np.array(f1)

    classAcc = np.sum(accuracy, axis=0)/batchRep
    classF1 = np.sum(f1, axis=0)/batchRep

    overallAcc = np.mean(classAcc)
    overallF1 = np.mean(classF1)

    with open(os.path.join(LOG_DIR, TEST_LOG), 'w') as f:
        f.write(f"F1 Score: {overallF1}\n")
        for idx, ff in enumerate(classF1):
            f.write(f"F1 for class {idx}: {ff}\n")
        f.write("==========================================\n")
        f.write(f"Overall Accuracy: {overallAcc}\n")
        for idx, acc in enumerate(classAcc):
            f.write(f"Accuracy for class {idx}: {acc}\n")
        f.flush()

if __name__ == "__main__":
    test()
    # model = get_model(NUM_CLASSES).to(DEVICE)