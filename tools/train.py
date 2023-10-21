from typing import List, Dict, Union, Optional
import numpy as np
import torch.nn as nn
from tools.dataloaders import get_folder_dataloader, get_split_TOROdataset
from src.SaLS.utils import fit_pnn

def train_category(model: nn.Module, 
                   pnn_parameters: Dict, 
                   device: str,
                   name: str,
                   root: str,
                   objs: List[str],
                   batch_size: int,
                   num_workers: int,
                   data_transforms: Dict) \
                   -> Dict[str, Union[float, np.array]]:
    '''
    Training phase.
    One column is added to the PNN for each given object.
    '''
    print('\x1b[1;34;43m' + '>> Training new category: {} '.format(objs) + '\x1b[0m')

    dataloaders_train = get_folder_dataloader(root=root,
                                              objs=objs if type(objs) is list else [objs], 
                                              data_transforms=data_transforms,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              train=True)
    
    dataloaders_val = get_folder_dataloader(root=root,
                                              objs=objs if type(objs) is list else [objs], 
                                              data_transforms=data_transforms,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              train=False)


    # Train PNN 
    metrics = {}
    for dataloader_train, dataloader_val in zip(dataloaders_train,dataloaders_val):
        metrics = fit_pnn(model = model, 
                          dataloader_train = dataloader_train,
                          parameters = pnn_parameters,
                          device = device,
                          name = name,
                          dataloader_val = dataloader_val)

    return metrics

def train_TORO_seq(model: nn.Module, 
                   pnn_parameters: Dict, 
                   device: str,
                   name: str,
                   root: str,
                   objs: List[str],
                   batch_size: int,
                   num_workers: int,
                   data_transforms: Dict,
                   excluded_seqs: Optional[List[str]] = None,) \
                   -> Dict[str, Union[float, np.array]]:
   
    # Choose random sequences for training and testing
    train_dataloaders, val_dataloaders, _ = get_split_TOROdataset(dataset_path=root,
                                                                  tasks=objs,
                                                                  train_ratio=0.7,
                                                                  data_transforms=data_transforms,
                                                                  batch_size=batch_size,
                                                                  num_workers=num_workers,
                                                                  max_data_seq=20,
                                                                  randomize=True,
                                                                  exclude_seqs=excluded_seqs)
    # Train PNN 
    metrics = {}
    for dataloader_train, dataloader_val in zip(train_dataloaders, val_dataloaders):
        metrics = fit_pnn(model = model, 
                          dataloader_train = dataloader_train,
                          parameters = pnn_parameters,
                          device = device,
                          name = name,
                          dataloader_val = dataloader_val)

    return metrics