import argparse
import cv2
import gc
import numpy as np
import os
import shutil
import sys

def reorganize_dataset(args):

    # new folder to save masks
    reoganized_masks_dir = os.path.join(args.input_dir,'..','masks_reorganized')
    os.makedirs(reoganized_masks_dir, exist_ok=True)

    list_dir = os.listdir(args.input_dir)
    list_instances = []
    list_objs = []
    list_seqs = []
    for seq in list_dir:
        obj_dir = os.listdir(os.path.join(args.input_dir,seq))
        for obj in obj_dir:
            instance_dir = os.listdir(os.path.join(args.input_dir,seq,obj))
            for instance in instance_dir:
                list_objs.append(obj)
                list_seqs.append(seq)
                list_instances.append(instance)

    for obj, seq, instance in zip(list_objs, list_seqs, list_instances):
        new_seq = seq.split('_')[1]
        new_path = os.path.join(reoganized_masks_dir,obj,new_seq)
        os.makedirs(new_path, exist_ok=True)

        new_instance = obj + '_' + new_seq + '_' + instance.split('_')[-1]
        shutil.copy(os.path.join(args.input_dir,seq,obj,instance), os.path.join(new_path,new_instance))

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(prog='Continual Semantic Segmentation',
                                     description='Description of the program',
                                     epilog='End of the program description')
    
    parser.add_argument('--input_dir', required=True, type=str, help='directory with input folder')
    args = parser.parse_args()
    
    reorganize_dataset(args)    