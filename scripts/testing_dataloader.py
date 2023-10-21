import argparse
import cv2
import gc
import json
import numpy as np
import os
import shutil
import sys
import torch

from PIL import Image
from tools.segmentation import get_center_obj_id

from tools.dataloaders import TORODataset

cat_seq = {'dark_mat': [1],
           'lab_floor': [2,3],}

a = TORODataset(root='/mnt/d/Documentos/DLR/Master_thesis/Thesis/Datasets/annotation_minarea8000/masks_reorganized',
                cat_seq=cat_seq,
                split='train')
print(a.categories)
print(a.classes)
print(a.sequences)
print(a.data)