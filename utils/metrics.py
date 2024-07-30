import torch
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from config import *

def f1Score(prediction, groundTruth):
    """
    Generate the F1 score given prediction and groundTruth per batch
    Args:
       Prediction: The model prediction for the example
       groundTruth: The true label for the example
    Returns:
       A numerical value summarizing the F1 score
    """
    preds = torch.argmax(prediction, dim=1)
    label_class = torch.argmax(groundTruth, dim = 1)
    return f1_score(preds.cpu(), label_class.cpu(), average='macro')

def f1ScorePerClass(prediction, groundTruth):
    """
    Generate the F1 score given prediction and groundTruth per batch
    Args:
       Prediction: The model prediction for the example
       groundTruth: The true label for the example
    Returns:
       A numerical value summarizing the F1 score
    """
    preds = torch.argmax(prediction, dim=1)
    label_class = torch.argmax(groundTruth, dim = 1)
    classF1 = f1_score(preds.cpu(), label_class.cpu(), labels=np.arange(7), average=None, zero_division=0)
    return classF1


def overallAccuracy(prediction, groundTruth):
    """
    Generate the accuracy given prediction and groundTruth per batch
    Args:
       Prediction: The model prediction for the example
       groundTruth: The true label for the example
    Returns: 
       A numerical value in % of accuracy for all classes
    """
    preds = torch.argmax(prediction, dim=1)
    label_class = torch.argmax(groundTruth, dim = 1)
    acc = torch.sum(preds == label_class) / len(preds)
    return acc

def classAccuracy(prediction, groundTruth):
    """
    Generate the accuracy for each class given prediction and groundTruth per batch
    Args:
       Prediction: The model prediction for the example
       groundTruth: The true label for the example
    Returns: 
       An array containing the percentage predicted correct for each class and which class is represented
   """

    labelTruth = torch.argmax(groundTruth, dim=1)
    predictions = torch.argmax(prediction, dim=1)

    labels = labelTruth.cpu().detach().numpy()
    preds = predictions.cpu().detach().numpy()

    occurence = np.bincount(labels, minlength=NUM_CLASSES)
    rep = np.nonzero(occurence)

    correct = np.where(labels == preds, preds, -1)
    correct = correct[correct >= 0]
    correct = np.bincount(correct, minlength=NUM_CLASSES)

    class_accuracy = np.divide(correct, occurence, out=np.zeros_like(correct, dtype=float), where=occurence!=0)
    return class_accuracy, rep[0]


def roc_auc_score_multiclass(prediction, groundTruth, average = "macro"):
    """
    Generate the Area Under the Curve for each class given prediction and groundTruth per batch
    Args:
       Prediction: The model prediction for the example
       groundTruth: The true label for the example
    Returns: 
       A dictionary containing the area under the curve for each class
    """
    #creating a set of all the unique classes using the actual class list
    unique_class = set(groundTruth)
    roc_auc_dict = {}
    for per_class in unique_class:
        
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in groundTruth]
        new_pred_class = [0 if x in other_class else 1 for x in prediction]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


#getSaliency map function still needed


if __name__ == "__main__":
   predictions = torch.tensor([[0.1, 0.2, 0.6, 0.1],
                              [0.7, 0.1, 0.1, 0.1],
                              [0.2, 0.2, 0.2, 0.4],
                              [0.1, 0.2, 0.6, 0.1]])

   ground_truth = torch.tensor([[0, 0, 1, 0],
                              [1, 0, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1]])
   
   b, rep = classAccuracy(predictions, ground_truth)

