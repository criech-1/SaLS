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

sys.path.append('./Segment-and-Track-Anything')
sys.path.append('./Segment-and-Track-Anything/aot')
from SegTracker import SegTracker
from model_args import aot_args,sam_args

def annotate_frames(args):
    print('\x1b[1;37;47m' + '* DATASET ANNOTATION *' + '\x1b[0m')
    print('*'*120)

    # Set torch seed
    torch.manual_seed(1)

    # Input/output directories
    io_args = {
        'input_dir': f'{args.input_dir}',
    }

    ### SAM-Track parameters ###################################################################

    min_area = args.min_area
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

    # aot_args['model'] = 'deaotl'
    # aot_args['model_path'] = './Segment-and-Track-Anything/ckpt/DeAOTL_PRE_YTB_DAV.pth'
    aot_args['model'] = 'r50_deaotl'
    aot_args['model_path'] = './Segment-and-Track-Anything/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth'

    # sam_args['sam_checkpoint'] = "./Segment-and-Track-Anything/ckpt/sam_vit_l_0b3195.pth"
    # sam_args['model_type'] = "vit_l"

    sam_args['sam_checkpoint'] = "./Segment-and-Track-Anything/ckpt/sam_vit_h_4b8939.pth"
    sam_args['model_type'] = "vit_h"

    ### Generate SAM-Track results #############################################################

    # Dict with object masks and labels
    masked_labels_pred = {} # Dict[int,label]    

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

    # Masks folder
    parent_folder = io_args['input_dir'].split('/')[-1]
    masks_dir = os.path.join(args.input_dir,'..','masks','{}_masks'.format(parent_folder))
    # Clean masks folder if exists
    if os.path.exists(masks_dir):
        shutil.rmtree(masks_dir)
    
    os.makedirs(masks_dir, exist_ok=True)

    # Initialize annotation dict
    annotation_dict = {} # {frame_idx: {obj: label,
                         #              bbox: [x,y,w,h],
                         #              point_prompt: [x,y]},
                         #              save: bool,}, ... }
    previous_label = ''
    previous_save = False
    new_object = False
    do_break = False

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

            ### Point segmentation mode
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
                new_object = True

            else: 
                # Track objects
                pred_mask = segtracker.track(frame,update_memory=True)
                new_object = False
            
            # Empty cache
            torch.cuda.empty_cache()
            gc.collect()

            id = 0
            # If there are objects in the frame
            if len(np.unique(pred_mask)) > 1:
                id = np.unique(pred_mask)[1]

                if id not in masked_labels_pred:
                    # Find object bounding box
                    crop_mask = (pred_mask == id).astype(np.uint8)
                    x,y,w,h = cv2.boundingRect(crop_mask)
                    crop_objs = [x,y,w,h]

                # Get center object
                center_id = get_center_obj_id(pred_mask, center)

                # Draw masks
                masked_frame = frame.copy()

                crop_mask = (pred_mask == id).astype(np.uint8)
                color = (0,200,0) # green in BGR

                # Fill with alpha=0.5 the object in the frame with the color
                masked_frame = cv2.addWeighted(masked_frame, 1, np.dstack((crop_mask*color[0],crop_mask*color[1],crop_mask*color[2])), 0.5, 0)
                # Draw contour of the object in the frame
                contours, _ = cv2.findContours(crop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(masked_frame, contours, -1, color, 2)

            # Blue cross in centre of frame
            cv2.line(masked_frame, (center[0]-10,center[1]), (center[0]+10,center[1]), (255, 0, 0), 2)
            cv2.line(masked_frame, (center[0],center[1]-10), (center[0],center[1]+10), (255, 0, 0), 2)

            frame_display = np.hstack([frame, masked_frame, cv2.cvtColor(pred_mask*255, cv2.COLOR_GRAY2BGR)]) 

            # Show frames in display
            cv2.rectangle(frame_display, (0,0), (250,40), (255,255,255), -1)
            cv2.putText(frame_display, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(frame_display, (width,0), (250+width,40), (255,255,255), -1)
            cv2.putText(frame_display, 'Segmentation', (width+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imwrite('dumb.png', frame_display)

            # Area of segmented object
            mask_area = np.sum(pred_mask == center_id)

            if new_object:
                while True:
                    input_ann = input('>> Frame {} with area {} [label save]: '.format(frame_idx, mask_area))
                    
                    try:
                        input_label, save_input = input_ann.split(' ')
                    except:
                        print('\x1b[1;37;41m' + 'Wrong input' + '\x1b[0m')
                        continue

                    if save_input in ['y','n']:
                        break
                    else:
                        print('\x1b[1;37;41m' + 'Wrong input' + '\x1b[0m')
                        
                save = True if save_input == 'y' else False

                previous_label = input_label
                previous_save = save
            else:
                input_label = previous_label
                save = previous_save

            if save:
                crop = frame * (pred_mask == id).astype(np.uint8)[...,None]
                crop = crop[y:y+h,x:x+w]

                if not os.path.exists(os.path.join(masks_dir, input_label)):
                    os.makedirs(os.path.join(masks_dir, input_label), exist_ok=True)

                cv2.imwrite(os.path.join(masks_dir, input_label, '{}_{}.png'.format(input_label,frame_idx)), crop)

            annotation_dict[frame_idx] = {}
            annotation_dict[frame_idx]['obj'] = input_label
            annotation_dict[frame_idx]['bbox'] = crop_objs
            annotation_dict[frame_idx]['point_prompt'] = center
            annotation_dict[frame_idx]['save'] = save
            
            # Press Q to exit
            if do_break:
                break

            print("Processed frame {}\t obj_label '{}'\t area {}\t saved '{}' ".format(frame_idx,annotation_dict[frame_idx]['obj'],mask_area,save))
            frame_idx += 1

        print("Finished {}\t obj_label '{}'\t area {}\t saved '{}' ".format(frame_idx,annotation_dict[frame_idx-1]['obj'],mask_area,save))

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

    # Save annotation dict in upper level from parent directory
    with open(os.path.join(args.input_dir,'..','masks','annotation_{}.json'.format(parent_folder)), 'w') as fp:
        json.dump(annotation_dict, fp, indent=4)

    print('\x1b[1;37;42m' + '>> Finished!' + '\x1b[0m')

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(prog='Continual Semantic Segmentation',
                                     description='Description of the program',
                                     epilog='End of the program description')
    
    parser.add_argument('--input_dir', required=True, type=str, help='directory with input images')
    parser.add_argument('--min_area', required=False, default=3500, type=int, help='min area for new segmented objects')
    args = parser.parse_args()
    
    annotate_frames(args)    