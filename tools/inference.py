import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from typing import List, Dict, Union, Optional
from torch.utils.data import DataLoader

def predict_dataloader(model: nn.Module,
                       dataloader: DataLoader,
                       device: str,
                       sigma: Optional[int] = None):
    # print('>> Dataloader class: ', dataloader.dataset.classes)
    pred_labels = []
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        pred_label, value = predict_maxlogit(inputs, model, sigma=sigma)
        pred_labels.append(pred_label)

    return pred_labels

def predict_masked_labels(model: nn.Module, 
                          img_dict: Dict[int, np.array],
                          masked_labels_pred: Dict[int, str],
                          data_transforms: Dict[str, torch.Tensor],
                          device: str,
                          sigma: int=2,):
    
    infer_batch = torch.stack([data_transforms['test'](img) for img in img_dict.values()])
    
    pred_labels = []
    for img in infer_batch:
        label, logit_max = predict_maxlogit(img.unsqueeze(0).to(device), model, sigma=sigma)
        pred_labels.append(label)

    for i, id in enumerate(img_dict.keys()):
        if id not in masked_labels_pred.keys():
            masked_labels_pred[id] = pred_labels[i]
        else:
            if masked_labels_pred[id] == 'Unknown':
                masked_labels_pred[id] = pred_labels[i]

    return masked_labels_pred


def predict_labels_point(model: nn.Module, 
                          img: Image,
                          data_transforms: Dict[str, torch.Tensor],
                          device: str,
                          sigma: int=2,):
    
    infer_img = data_transforms['test'](img)
    label, logit_max = predict_maxlogit(infer_img.unsqueeze(0).to(device), model, sigma=sigma)

    # if frame_idx not in masked_labels_pred.keys():
    #     masked_labels_pred[frame_idx] = label

    return label

def predict_maxlogit(inputs: torch.Tensor,
                     model: nn.Module,
                     sigma: int):

    sigma = sigma if sigma is not None else 2

    model.eval()

    with torch.no_grad():
        logits = model(inputs)

        preds = []
        for logit in logits:
            preds.append(logit.cpu().numpy()[0][0])

    pred_idx = np.argmin(preds,axis=0)
    pred_value = preds[pred_idx]

    mean_logit = model.networks_maxlogits[pred_idx][0]
    std_logit = model.networks_maxlogits[pred_idx][1]

    if pred_value <= mean_logit + sigma * std_logit:
        label = model.networks_labels[pred_idx][0]
    else:
        label = 'Unknown'
        
    return label, pred_value