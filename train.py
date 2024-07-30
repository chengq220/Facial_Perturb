import torch
from dataset import *
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from config import *
import os
from utils.util import *
from utils.metrics import f1Score, overallAccuracy
from torch.cuda.amp import GradScaler, autocast
from models.selection import selectModel
from models.clustering import AttentionClusterAugment

def train():
    model = selectModel()
    trainLoader, valLoader, _ = getDataloaders()

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay = DECAY, momentum=MOMENT, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5,verbose=True)
    cluster = AttentionClusterAugment()


    with open(os.path.join(LOG_DIR, TRAIN_LOG), 'w') as f:
        ep = 1
        # if RESUME:
        #     modelState, optimState, epoch = load_network()
        #     f.write(f'Loading model and optim from epoch {epoch}\n')
        #     ep = epoch
        #     model.load_state_dict(modelState)
        #     optimizer.load_state_dict(optimState)
        #     f.flush()
    
        total_train_losses   = []
        total_val_losses     = []
        total_train_accuracy = []
        total_val_accuracy   = []
        total_train_f1       = []
        total_val_f1         = []

        for epoch in range(ep, EPOCHS+1):
            model.train()
            train_losses   = []
            train_accuracy = []
            train_f1       = []
            f.write(f"Epoch: {epoch}\n")
            clusterUpdate = True
            for i, batch in enumerate(tqdm(trainLoader, desc=f"Epoch {epoch}")):
                #Extract data, labels
                img_batch, label_batch = batch   #img [B,3,H,W], label[B,N_CLASSES]
                img_batch = img_batch.to(DEVICE)
                label_batch = label_batch.to(DEVICE)
                bs, ncrops, c, h, w = img_batch.shape
                img_batch = img_batch.view(-1, c, h, w)
                label_batch = torch.repeat_interleave(label_batch, repeats=ncrops, dim=0)
                img_batch = cluster(img_batch, update = clusterUpdate)
                l_b, l_o = label_batch.shape
                label_batch = label_batch.expand(HEADS+1, l_b, l_o).reshape((HEADS+1) * l_b,l_o)
                clusterUpdate = False

                #Train model
                optimizer.zero_grad()
                output = model(img_batch) # output: [B, 7, H, W]  #VGG
                loss   = criterion(output, label_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                #Add current loss to temporary list (after 1 epoch take avg of all batch losses)
                f1 = f1Score(output, label_batch)
                acc = overallAccuracy(output, label_batch)
                train_losses.append(loss.item())
                train_accuracy.append(acc.cpu())
                train_f1.append(f1)
        
            #calculate the metric for 1 epoch
            total_train_losses.append(np.mean(train_losses))
            total_train_accuracy.append(np.mean(train_accuracy))
            total_train_f1.append(np.mean(train_f1))
            
            f.write(f"Train Loss: {total_train_losses[-1]}\n")
            f.write(f"Train Accuracy: {total_train_accuracy[-1]}\n")
            f.write(f"Train F1 : {total_train_f1[-1]}\n")
            f.flush()
            
            # Save network
            save_network({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            }) 

            model.eval()
            val_losses   = []
            val_accuracy = []
            val_f1       = []

            for i, batch in enumerate(tqdm(valLoader)):
                #Extract data, labels
                img_batch, label_batch = batch   #img [B,3,H,W], label[B,N_CLASSES]
                img_batch = img_batch.to(DEVICE)
                label_batch = label_batch.to(DEVICE)
                bs, ncrops, c, h, w = img_batch.shape
                img_batch = img_batch.view(-1, c, h, w)
                img_batch = cluster(img_batch)
                label_batch = torch.repeat_interleave(label_batch, repeats=ncrops, dim=0)
                print(img_batch.shape)
                print(label_batch.shape)

                #Validate model
                with torch.cuda.amp.autocast():
                    output = model(img_batch) # output: [B, 7, H, W] #for vgg
                    loss   = criterion(output, label_batch)

                #Add current loss to temporary list (after 1 epoch take avg of all batch losses)
                f1 = f1Score(output, label_batch)
                acc = overallAccuracy(output, label_batch)
                val_losses.append(loss.item())
                val_accuracy.append(acc.cpu())
                val_f1.append(f1)

            # #the average of the metric for the testing 
            total_val_losses.append(np.mean(val_losses))
            total_val_accuracy.append(np.mean(val_accuracy))
            total_val_f1.append(np.mean(val_f1))  

            # Step the learning rate
            scheduler.step(torch.tensor(np.mean(val_accuracy)))

            f.write(f"Val Loss: {total_val_losses[-1]}\n")
            f.write(f"Val Accuracy: {total_val_accuracy[-1]}\n")
            f.write(f"Val F1 : {total_val_f1[-1]}\n")
            f.write("======================================================================\n")
            f.flush()


if __name__ == "__main__":
    train()