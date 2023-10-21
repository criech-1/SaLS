import argparse
import cv2
import gc
import json
import numpy as np
import os
import shutil
import sys
import torch
import torchvision.transforms as transforms

from copy import deepcopy

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import pandas as pd

from src.SaLS.pnn import ProgressiveNeuralNetwork, ClassifierNet
from tools.train import train_category, train_TORO_seq
from tools.inference import predict_masked_labels
from tools.segmentation import save_mask_objects, get_center_obj_id, draw_masks_labels, draw_center_label
from tools.dataloaders import create_folder_dataset, TORO_cats_seqs

from src.SaLS.utils import seed_all_rng

sys.path.append('./Segment-and-Track-Anything')
sys.path.append('./Segment-and-Track-Anything/aot')
from SegTracker import SegTracker
from model_args import aot_args,sam_args

main_dir = '/u/52/echevec1/unix/toro/TORO_dataset'

# Tasks to evaluate
tasks = ['dark_mat', 'lab_floor', 'stones', 'ramp', 'blue_mattress', 'pedestal']
num_tasks = len(tasks)

def sequences_metrics(args):
    print('\x1b[1;37;47m' + '* SEQUENCES METRICS *' + '\x1b[0m')
    print('*'*104)

    # Use cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dir = os.path.join(main_dir,'sequences', str(args.seq))

    # Input/output directories
    io_args = {
        'input_dir': input_dir,
        'output_mask_dir': f'{input_dir}_masks_point_{args.id}', # object masks
        'output_dir': f'{input_dir}_output_point_{args.id}', # mask+frame
    }
    ### SAM-Track parameters ###################################################################

    min_area = 8000
    # SAM-Track parameters
    segtracker_args = {
        'sam_gap': 9999,        # the interval to run sam to segment new objects (only auto mode)
        'min_area': min_area,   # minimal mask area to add a new mask as a new object
        'max_obj_num': 255,     # maximal object number to track in a video
        'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
    }

    # SAM parameters
    sam_args['generator_args'] = {
            'points_per_side': 32,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': min_area, #200,
        }
    
    aot_args['model'] = 'r50_deaotl'
    aot_args['model_path'] = './Segment-and-Track-Anything/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth'

    sam_args['sam_checkpoint'] = "./Segment-and-Track-Anything/ckpt/sam_vit_h_4b8939.pth"
    sam_args['model_type'] = "vit_h"

    ### Initialize DINO and PNN ################################################################

    # Load DINOv2 model from torch hub or local path if net not available
    try:
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
        print('\x1b[1;37;42m' + '>> DINOv2 model loaded from hub' + '\x1b[0m')
    except: 
        dino_model = torch.hub.load('dinov2', 'dinov2_vits14', source='local', pretrained=False)
        dino_model.load_state_dict(torch.load('./Segment-and-Track-Anything/ckpt/dinov2_vits14_pretrain.pth'))
        dino_model.to(device).eval()
        print('\x1b[1;37;42m' + '>> DINOv2 model loaded locally' + '\x1b[0m')

    print('*'*120)

    # Load base network
    network = ClassifierNet(input_size = list(dino_model.children())[-2].normalized_shape[0],
                            output_size=1) # 384 if DINOv2 with ViT-Small
    base_network = deepcopy(network).to(device)

    # Initialize PNN
    last_layer_name = 'fc'
    lateral_connections = None

    pnn_model = ProgressiveNeuralNetwork(base_network = base_network,
                                         backbone = dino_model,
                                         last_layer_name = last_layer_name,
                                         lateral_connections = deepcopy(lateral_connections))

    # PNN parameters
    parameters = {
        'learning_rate': 0.001,
        'weight_decay': 1e-8,
        'patience': 20,
        'num_epochs': 50,
    }

    ### Train initial PNN ######################################################################

    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(280), # try with other number, apparently in ? paper they use 518
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(312),
            transforms.CenterCrop(280),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(312),
            transforms.CenterCrop(280),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Check if PNN is already trained
    pnn_dir = '/u/52/echevec1/unix/toro/SaLS/models/exp_seq{}_{}/full_state_dict_6.pt'.format(args.seq, args.id)
    if os.path.exists(pnn_dir):
        pnn_model.load_full_state_dict(torch.load(pnn_dir))
        pnn_model.to(device)
    else:
        # Initial PNN training
        init_train_metrics = train_TORO_seq(model=pnn_model, 
                                            pnn_parameters=parameters, 
                                            device=device,
                                            name='exp_seq{}_{}'.format(args.seq, args.id),
                                            root=args.train_dir,
                                            objs=tasks, 
                                            batch_size=1,
                                            num_workers=4,
                                            data_transforms=data_transforms,
                                            excluded_seqs=[str(args.seq)])

    # Empty cache
    torch.cuda.empty_cache()
    gc.collect() 
    
    print('\x1b[1;37;42m' + '>> PNN model initialized and trained' + '\x1b[0m')
    print(' >> Learnt labels: ', pnn_model.networks_labels)

    ### Generate SAM-Track results #############################################################

    # Output masks
    output_crop = io_args['output_mask_dir']
    # Clean cropped objs directory if exists
    shutil.rmtree(output_crop, ignore_errors=True)
    os.makedirs(output_crop, exist_ok=True)

    # Output dir
    shutil.rmtree(io_args['output_dir'], ignore_errors=True)
    os.makedirs(io_args['output_dir'], exist_ok=True)

    # Dict with object masks and labels
    masked_labels_pred = {} # Dict[int,label] = {frame_idx: label}

    # Empty cache
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize SegTracker
    frame_idx = 0
    segtracker = SegTracker(segtracker_args,sam_args,aot_args)
    segtracker.restart_tracker()

    print('*'*104)

    # Initialize time counter
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # If folder, order frames
    frames_path_dir = os.listdir(io_args['input_dir'])
    frames_path_dir.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    num_frames = len(frames_path_dir)
    print('\x1b[1;37;43m' + '>> Reading frames from folder...' + '\x1b[0m')

    # Results dict
    results_dict = {} # {frame_idx: {label: str,
                      #              train: bool, }}

    with torch.cuda.amp.autocast(True):
        while True:
            # If folder, check if last frame
            if frame_idx == num_frames:
                break
            else:
                # Read frame from folder
                frame = cv2.imread(os.path.join(io_args['input_dir'],frames_path_dir[frame_idx]))

            # Set frame characteristics in first frame
            if frame_idx == 0:
                height, width, channels = frame.shape
                # Find center of frame
                center = (int(width / 2), int(height / 2))
                center_track = np.array([[center[0],center[1]]])
  
            # Get center id, 0 if first frame
            center_id = 0 if frame_idx == 0 else get_center_obj_id(pred_mask, center)

            # If center is not segmented
            if center_id == 0:
                # Restart tracker and pred labels
                masked_labels_pred = {} 
                segtracker.restart_tracker()

                # Run prompt segmentation
                segtracker.sam.interactive_predictor.set_image(frame)
                pred_mask = segtracker.sam.segment_with_click(frame,center_track,np.array([1]),segtracker_args['min_area'],multimask=True)
                
                # Empty cache
                torch.cuda.empty_cache()
                gc.collect()

                # Add objects to tracker
                segtracker.add_reference(frame, pred_mask)

                trained = False
            else: 
                # Track objects
                pred_mask = segtracker.track(frame,update_memory=True)
                center_id = get_center_obj_id(pred_mask, center)

                # If center is not segmented
                if center_id == 0:
                    # Restart tracker and pred labels
                    masked_labels_pred = {} 
                    segtracker.restart_tracker()

                    # Run prompt segmentation
                    segtracker.sam.interactive_predictor.set_image(frame)
                    pred_mask = segtracker.sam.segment_with_click(frame,center_track,np.array([1]),segtracker_args['min_area'],multimask=True)
                    
                    # Empty cache
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Add objects to tracker
                    segtracker.add_reference(frame, pred_mask)

                    trained = False

            # Empty cache
            torch.cuda.empty_cache()
            gc.collect()

            # Crop and save objects
            crop_objs = save_mask_objects(pred_mask, frame, output_crop, frame_idx, masked_labels_pred) # dict = {id: Image}

            # Predict labels
            if bool(crop_objs): # if dict is not empty
                masked_labels_pred = predict_masked_labels(model=pnn_model, 
                                                           img_dict=crop_objs,
                                                           masked_labels_pred=masked_labels_pred,
                                                           data_transforms=data_transforms,
                                                           device=device,)

            # Get center object
            center_id = get_center_obj_id(pred_mask, center)

            # Train PNN with unknown category/ies: check if center object is unknown and CL arg is True
            if center_id != 0 and args.cl:
                # Check if center object is unknown and not in skipped_masks
                if masked_labels_pred[center_id] == 'Unknown':
                    print('\nCenter object in frame {} is not in the training set.'.format(frame_idx))
                    # Ask for new label
                    new_category = input('\x1b[1;37;45m' + 'Write new label for training:' + '\x1b[0m' + ' ')

                    # Create folder dataset
                    create_folder_dataset(output_crop=output_crop,
                                          new_category=new_category,
                                          obj_id=center_id,
                                          frame_idx=frame_idx,
                                          frame=frame,
                                          pred_mask=pred_mask)

                    # Train PNN with new object
                    with torch.cuda.amp.autocast(False):
                        metrics = train_category(model=pnn_model, 
                                                 pnn_parameters=parameters, 
                                                 device=device,
                                                 name=None, #'exp_seq{}_{}'.format(args.seq, args.id),
                                                 root=output_crop,
                                                 objs=new_category, 
                                                 batch_size=1,
                                                 num_workers=4,
                                                 data_transforms=data_transforms)
                        trained = True

                    if bool(crop_objs): # if dict is not empty
                        masked_labels_pred = predict_masked_labels(model=pnn_model, 
                                                                    img_dict=crop_objs,
                                                                    masked_labels_pred=masked_labels_pred,
                                                                    data_transforms=data_transforms,
                                                                    device=device,)
                    # Check if still unknown
                    if masked_labels_pred[center_id] == 'Unknown':
                        print('\x1b[1;37;41m' + '>> Training not OK.' + '\x1b[0m')
                        masked_labels_pred[center_id] = new_category

                    print('\x1b[1;37;42m' + '>> Center object in frame {} new label: {}'.format(frame_idx, masked_labels_pred[center_id]) + '\x1b[0m')

            # Draw masks
            masked_frame = frame.copy()
            masked_frame = draw_masks_labels(masked_frame, pred_mask, masked_labels_pred)
            masked_frame = draw_center_label(masked_frame, center, masked_labels_pred, center_id)
            frame_display = np.hstack([frame, masked_frame]) 

            # Write frame in output video
            # if args.id == 0:
            cv2.imwrite(os.path.join(io_args['output_dir'], str(frame_idx)+'.png'), frame_display)

            # check if masked_labels_pred[frame_idx] exists
            if center_id not in masked_labels_pred.keys():
                masked_labels_pred[center_id] = 'Not segmented'
                trained = False

            results_dict[frame_idx] = {}
            results_dict[frame_idx] = {'label': masked_labels_pred[center_id],
                                       'train': trained}

            print("Processed frame {}, obj_num {}, center_id {}".format(frame_idx,segtracker.get_obj_num(),center_id),end='\r')
            frame_idx += 1

        print('Finished {}, obj_num {}, center_id {}.'.format(frame_idx,segtracker.get_obj_num(),center_id))

    print('Masked objects: ' + str(masked_labels_pred))
    cv2.waitKey()
    cv2.destroyAllWindows()

    end.record()
    torch.cuda.synchronize()
    end_time = np.round(start.elapsed_time(end)/1000,2)
    print('Total time: {} s.'.format(end_time))

    # Manually release memory (after cuda out of memory)
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()

    # Save results dict in upper level from parent directory
    with open('/u/52/echevec1/unix/toro/TORO_dataset/results/results_seq{}_{}.json'.format(args.seq, args.id), 'w') as fp:
        json.dump(results_dict, fp, indent=4)

    print('\x1b[1;37;42m' + '>> Finished!' + '\x1b[0m')

def sequences_evaluation(args):
    print('\x1b[1;37;47m' + '* SEQUENCES EVALUATION *' + '\x1b[0m')
    print('*'*120)

    input_dir = os.path.join(main_dir,'sequences', str(args.seq))

    # Input/output directories
    io_args = {
        'input_dir': input_dir,
        'output_mask_dir': f'{input_dir}_masks_point_{args.id}', # object masks
        'output_dir': f'{input_dir}_output_point_{args.id}', # mask+frame
    }

    # Load gt dict
    with open(os.path.join(main_dir,'annotations','annotation_seq{}.json'.format(args.seq)), 'r') as fp:
        gt_dict = json.load(fp)

    gt_labels = []
    for frame in gt_dict:
        gt_labels.append(gt_dict[frame]['obj'])

    results_folder = os.listdir(os.path.join(main_dir,'results'))
    results_list = []
    results_str = 'results_seq{}'.format(args.seq)
    for result in results_folder:
        if results_str in result:
            results_list.append(result)

    categories = []
    num_correct = []
    num_total = []

    for result in results_list:
        with open(os.path.join(main_dir,'results', result), 'r') as fp:
            res_dict = json.load(fp)

        for frame in res_dict:
            if frame != list(gt_dict.keys())[-1]:
                if gt_dict[str(int(frame)+1)]['obj'] in TORO_cats_seqs.keys():
                    if gt_dict[str(int(frame)+1)]['obj'] not in categories:
                        categories.append(gt_dict[str(int(frame)+1)]['obj'])
                        num_correct.append(0)
                        num_total.append(0)

                    cat_idx = categories.index(gt_dict[str(int(frame)+1)]['obj'])

                    if res_dict[frame]['label'] == gt_dict[str(int(frame)+1)]['obj']:
                        num_correct[cat_idx] += 1
                    num_total[cat_idx] += 1

    idx_known = []
    idx_unknown = []

    for category in categories:
        if category in tasks:
            idx_known.append(categories.index(category))
        else:
            idx_unknown.append(categories.index(category))

    print('Accuracy known:\t', 100*np.sum(np.array(num_correct)[idx_known]) / np.sum(np.array(num_total)[idx_known]))
    print('Accuracy unknown:\t', 100*np.sum(np.array(num_correct)[idx_unknown]) / np.sum(np.array(num_total)[idx_unknown]))
    print('Average accuracy:\t', 100*np.sum(num_correct) / np.sum(num_total))

    print('-'*30)

    for category in categories:
        print(category,'\t', 100*num_correct[categories.index(category)] / num_total[categories.index(category)])

def sequences_evaluation_cm():
    sequences = [1,2,3,4,5,6,7]

    input_dir = os.path.join(main_dir,'sequences', str(args.seq))

    # Input/output directories
    io_args = {
        'input_dir': input_dir,
        'output_mask_dir': f'{input_dir}_masks_point_{args.id}', # object masks
        'output_dir': f'{input_dir}_output_point_{args.id}', # mask+frame
    }

    y_true = []
    y_pred = []

    results_folder = os.listdir(os.path.join(main_dir,'results'))

    categories = list(TORO_cats_seqs.keys())
    cat_idx = list(range(len(categories)))

    for seq in sequences:
        # Load gt dict
        with open(os.path.join(main_dir,'annotations','annotation_seq{}.json'.format(seq)), 'r') as fp:
            gt_dict = json.load(fp)

        results_list = []
        results_str = 'results_seq{}'.format(seq)
        for result in results_folder:
            if results_str in result:
                results_list.append(result)

        for result in results_list:
            with open(os.path.join(main_dir,'results', result), 'r') as fp:
                res_dict = json.load(fp)

            for frame in res_dict:
                if frame != list(gt_dict.keys())[-1]:
                    if gt_dict[str(int(frame)+1)]['obj'] in TORO_cats_seqs.keys() and res_dict[frame]['label'] in TORO_cats_seqs.keys():

                        gt_idx = cat_idx[categories.index(gt_dict[str(int(frame)+1)]['obj'])]
                        pred_idx = cat_idx[categories.index(res_dict[frame]['label'])]

                        y_true.append(gt_idx)
                        y_pred.append(pred_idx)

    cm = confusion_matrix(y_true, y_pred, labels=cat_idx)
    cm_norm = confusion_matrix(y_true, y_pred, labels=cat_idx, normalize='true')

    df_cm = pd.DataFrame(cm, index = categories, columns = categories)
    df_cm_norm = pd.DataFrame(cm_norm*100, index = categories, columns = categories)

    plt.figure(figsize = (10,10))
    sn.heatmap(df_cm, annot=True, cmap='Blues',fmt='g')
    plt.savefig('cm.png')

    plt.figure(figsize = (10,10))
    sn.heatmap(df_cm_norm, annot=True, cmap='Blues', fmt='.1f')
    plt.savefig('cm_norm.png')

    # total accuracy
    print('Accuracy:\t', 100*accuracy_score(y_true, y_pred))

    # accuracy of known categories
    idx_known = []
    idx_unknown = []

    for category in categories:
        if category in tasks:
            idx_known.append(categories.index(category))
        else:
            idx_unknown.append(categories.index(category))

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(prog='Continual Semantic Segmentation',
                                     description='Description of the program',
                                     epilog='End of the program description')
    
    parser.add_argument('--id', type=int, required=True, help='experiment id and seed')
    parser.add_argument('--seq', type=int, required=True, help='sequence to run')
    parser.add_argument('--train_dir', required=False, type=str, help='directory with training images, if no --load_pnn')
    parser.add_argument('--cl', action='store_true', help='use PNN with CL')
    parser.add_argument('--no-cl', dest='cl', action='store_false', help='use PNN as simple classifier')
    parser.set_defaults(cl=False)
    parser.add_argument('--eval', action='store_true', help='evaluate PNN with existing metrics')
    parser.add_argument('--no-eval', dest='eval', action='store_false', help='get PNN metrics')
    parser.set_defaults(eval=False)
    args = parser.parse_args()

    # Set manual seed
    seed_all_rng(args.id)

    if args.eval:
        sequences_evaluation(args)
        print('*'*120)
        sequences_evaluation_cm()
    else:
        sequences_metrics(args)   