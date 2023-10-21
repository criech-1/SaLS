import argparse
import cv2
import gc
import numpy as np
import os
import shutil
import sys
import torch
import torchvision.transforms as transforms

from copy import deepcopy
from PIL import Image

from src.SaLS.pnn import ProgressiveNeuralNetwork, ClassifierNet
from tools.train import train_category
from tools.inference import predict_masked_labels
from tools.segmentation import save_mask_objects, get_center_obj_id, draw_masks_labels, draw_center_label
from tools.dataloaders import create_folder_dataset

sys.path.append('./Segment-and-Track-Anything')
sys.path.append('./Segment-and-Track-Anything/aot')
from SegTracker import SegTracker
from model_args import aot_args,sam_args

def acquire_image():
    data = sn_depth_channel.read(InChannel.BLOCKING_WAIT)
    if data is None:
        raise RuntimeError('Couldn\'t acquire data.')

    print("head", str(data.head))
    print("ts", str(data.ts))
    print("field0", type(data.field0), data.field0.shape, data.field0.strides)
    print("field1", type(data.field1), data.field1.shape, data.field1.strides)
    print("field2", type(data.field2), data.field2.shape, data.field2.strides)
    print("field3", type(data.field3), data.field3.shape, data.field3.strides)
    print("field0", "min=%d" % np.amin(data.field0), "max=%d" % np.amax(data.field0))
    return data.field2 if data is not None else None

def toro_SaLS_segmentation(args):
    print('\x1b[1;37;44m' + '*'*26 + '\x1b[0m')
    print('\x1b[1;37;44m' + '*' + '\x1b[0m' + '\033[1m' + ' TORO SaLS SEGMENTATION ' + '\033[0m' + '\x1b[1;37;44m' + '*' + '\x1b[0m')
    print('\x1b[1;37;44m' + '*'*26 + '\x1b[0m')
    print('*'*120)

    # Use cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Input/output directories
    io_args = {
        'input_dir': f'{args.input_dir}',
        'output_mask_dir': f'{args.input_dir}_masks_{args.mode}', # object masks
        'output_dir': f'{args.input_dir}_output_{args.mode}', # mask+frame
    }

    ### SAM-Track parameters ###################################################################

    min_area = 20000
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

    if args.load_pnn == None:
        # If not pre-trained, train with directory of images
        terrains_dir = args.train_dir
        # Specify classes to train
        terrains = os.listdir(terrains_dir) # ['blue_mat','dark_mat','grey_ramp','lab_floor','pedestal','stones']

        # Initial PNN training
        init_train_metrics = train_category(model=pnn_model, 
                                            pnn_parameters=parameters, 
                                            device=device,
                                            root=terrains_dir,
                                            objs=terrains, 
                                            batch_size=1,
                                            num_workers=4,
                                            data_transforms=data_transforms,)
        
        print('\x1b[1;37;42m' + '>> PNN model initialized and trained' + '\x1b[0m')
        print(' >> Learnt labels: ', pnn_model.networks_labels)
    else:
        # If pre-trained, load weights
        pnn_model.load_full_state_dict(torch.load(args.load_pnn))
        pnn_model.to(device)

        print('\x1b[1;37;42m' + '>> PNN model loaded' + '\x1b[0m')
        print(' >> Learnt labels: ', pnn_model.networks_labels)

    print('*'*120)

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
    masked_labels_pred = {} # Dict[int,label]
    # List with skipped objects (auto mode)
    skipped_masks = []

    # Empty cache
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize SegTracker
    sam_gap = segtracker_args['sam_gap']
    frame_idx = 0
    segtracker = SegTracker(segtracker_args,sam_args,aot_args)
    segtracker.restart_tracker()

    print('*'*120)

    # Initialize time counter
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    if args.cam:
        # If camera, initilize stream. For TORO:
        from pysn.stream import InChannel
        sn_depth_channel = InChannel('sn_depth')
        print('\x1b[1;37;43m' + '>> Reading frames from camera...' + '\x1b[0m')
        print('>> Reading frames from camera...')

    else:
        # If folder, order frames
        frames_path_dir = os.listdir(io_args['input_dir'])
        frames_path_dir.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        num_frames = len(frames_path_dir)
        print('\x1b[1;37;43m' + '>> Reading frames from folder...' + '\x1b[0m')

    with torch.cuda.amp.autocast(True):
        while True:
            if args.cam:
                # If camera, function to get frame
                frame = acquire_image()
            else:
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

            # Automatic segmentation mode
            if args.mode == 'auto':
                # Run segmentation every sam_gap frame
                if (frame_idx % sam_gap) == 0:
                    pred_mask = segtracker.seg(frame)
                    # Empty cache
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # If not first frame, track objects
                    if frame_idx != 0:
                        # Track objects
                        track_mask = segtracker.track(frame)
                        # Find new objects, and update tracker with new objects
                        new_obj_mask = segtracker.find_new_objs(track_mask,pred_mask)
                        pred_mask = track_mask + new_obj_mask

                    # Add objects to tracker
                    segtracker.add_reference(frame, pred_mask)

                else:
                    # Track objects
                    pred_mask = segtracker.track(frame,update_memory=True)

            # Point segmentation mode
            elif args.mode == 'point':
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
                if masked_labels_pred[center_id] == 'Unknown' and center_id not in skipped_masks:
                    print('\nCenter object {} is not in the training set.'.format(center_id))
                    # Ask for new label
                    new_category = input('\x1b[1;37;45m' + 'Write new label for training, or "skip" otherwise (auto mode):' + '\x1b[0m')

                    if new_category != 'skip':
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
                                                     name=args.name,
                                                     root=output_crop,
                                                     objs=new_category, 
                                                     batch_size=1,
                                                     num_workers=4,
                                                     data_transforms=data_transforms)
                        
                        # Inference in center object
                        crop_objs = {}
                        list_dir = os.listdir(output_crop)

                        obj_name = [s for s in list_dir if s.split('_')[0] == str(center_id)][0]
                        crop_objs[center_id] = Image.open(os.path.join(output_crop, obj_name))

                        if bool(crop_objs): # if dict is not empty
                            masked_labels_pred = predict_masked_labels(model=pnn_model, 
                                                                       img_dict=crop_objs,
                                                                       masked_labels_pred=masked_labels_pred,
                                                                       data_transforms=data_transforms,
                                                                       device=device,)

                        print('\x1b[1;37;42m' + '>> Center object {} new label: {}'.format(center_id, masked_labels_pred[center_id]) + '\x1b[0m')
                    else:
                        skipped_masks.append(center_id)
                        print('>> Skipped center object: ', center_id)

            # Draw masks
            masked_frame = frame.copy()
            masked_frame = draw_masks_labels(masked_frame, pred_mask, masked_labels_pred)
            masked_frame = draw_center_label(masked_frame, center, masked_labels_pred, center_id)
            frame_display = np.hstack([frame, masked_frame]) 

            # Show frames in display
            window_name = 'Video segmentation'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 2*width, height)
            cv2.imshow(window_name, frame_display)
            
            # Write frame in output video
            cv2.imwrite(os.path.join(io_args['output_dir'], str(frame_idx)+'.png'), frame_display)

            # Press Q to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

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

    print('\x1b[1;37;42m' + '>> Finished!' + '\x1b[0m')

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(prog='Continual Semantic Segmentation',
                                     description='Description of the program',
                                     epilog='End of the program description')
    
    parser.add_argument('--name', '-n', required=True, type=str, help='experiment name')
    parser.add_argument('--cam', action='store_true', help='use camera as input')
    parser.add_argument('--no-cam', dest='cam', action='store_false', help='use folder with frames as input')
    parser.set_defaults(cam=False)
    parser.add_argument('--cl', action='store_true', help='use PNN with CL')
    parser.add_argument('--no-cl', dest='cl', action='store_false', help='use PNN as simple classifier')
    parser.set_defaults(cl=False)
    parser.add_argument('--mode', required=True, type=str, default='auto', choices={'auto', 'point'}, help='choose SAM mode')
    parser.add_argument('--input_dir', required=True, type=str, help='directory with input images, if --no-cam')
    parser.add_argument('--load_pnn', required=False, type=str, default=None, help='directory with trained PNN')
    parser.add_argument('--train_dir', required=False, type=str, help='directory with training images, if no --load_pnn')
    args = parser.parse_args()
    
    toro_SaLS_segmentation(args)    