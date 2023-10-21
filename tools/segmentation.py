import cv2
import os
import numpy as np
from PIL import Image
from typing import Optional, Callable, Tuple, Any, Dict, List

def save_mask_objects(pred_mask: np.ndarray,
                      frame: np.ndarray,
                      output_dir: str, 
                      frame_id: int, 
                      masked_labels_pred: Dict[int, str]):
    crop_objs = {}
    # Loop through objects detected in mask, unique id
    for id in np.unique(pred_mask):
        if id != 0:
            if id not in masked_labels_pred:
                # Find object bounding box
                crop_mask = (pred_mask == id).astype(np.uint8)
                x,y,w,h = cv2.boundingRect(crop_mask)
                # Crop object from frame and save it
                crop = frame * (pred_mask == id).astype(np.uint8)[...,None]
                crop = crop[y:y+h,x:x+w]
                crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop.save(os.path.join(output_dir, str(id)+'_{}.png'.format(frame_id)))
                crop_objs[id] = crop
    # Return list of cropped objects
    return crop_objs # dict = {id: Image}

def get_center_obj_id(pred_mask: np.ndarray,
                      center: Tuple[int, int],):
    '''
    Returns the id of the object that contains the center point.
    '''
    center_id = pred_mask[center[1], center[0]]

    return center_id

def draw_masks(masked_frame: np.ndarray, 
               pred_mask: np.ndarray, 
               alpha: float = 0.5):
    for id in np.unique(pred_mask):
        if id != 0:
            color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

            crop_mask = (pred_mask == id).astype(np.uint8)
            # Fill with alpha the object in the frame with the color
            masked_frame = cv2.addWeighted(masked_frame, 1, np.dstack((crop_mask*color[0],crop_mask*color[1],crop_mask*color[2])), alpha, 0)
            # Draw contour of the object in the frame
            contours, _ = cv2.findContours(crop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(masked_frame, contours, -1, color, 1)
        else:
            # color = (0,0,0)
            continue

    return masked_frame

def draw_masks_labels(masked_frame: np.ndarray, 
                      pred_mask: np.ndarray, 
                      masked_labels_pred: Dict[int, str],
                      alpha: float = 0.5):
    
    for id in np.unique(pred_mask):
        crop_mask = (pred_mask == id).astype(np.uint8)

        if id in masked_labels_pred:
            if masked_labels_pred[id] == 'Unknown':
                # B G R
                color = (0,0,200) # red in BGR
            else: 
                color = (0,200,55) # green in BGR
        else:
            # blue in BGR
            color = (200,0,0) # blue in BGR

        # Fill with alpha=0.5 the object in the frame with the color
        masked_frame = cv2.addWeighted(masked_frame, 1, np.dstack((crop_mask*color[0],crop_mask*color[1],crop_mask*color[2])), alpha, 0)
        # Draw contour of the object in the frame
        contours, _ = cv2.findContours(crop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(masked_frame, contours, -1, color, 2)
    
    return masked_frame

def draw_center_label(masked_frame: np.ndarray,
                      center: Tuple[int, int],
                      masked_labels_pred: Dict[int, str],
                      id: int):

    # Blue cross in centre of frame
    cv2.line(masked_frame, (center[0]-10,center[1]), (center[0]+10,center[1]), (255, 0, 0), 2)
    cv2.line(masked_frame, (center[0],center[1]-10), (center[0],center[1]+10), (255, 0, 0), 2)

    if id in masked_labels_pred:
        label = masked_labels_pred[id]
        if label == 'Unknown':
            # B G R
            color = (0,0,200) # red in BGR
        else: 
            color = (0,200,55) # green in BGR
    else:
        # blue in BGR
        color = (200,0,0) # blue in BGR
        label = 'Not segmented'

    # Put text with label of center object above the circle
    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    textX = int((20 - textsize[0]) / 2)
    textY = int((20 + textsize[1]) / 2)
    # rectangle around text bigger at least 5 pixels than text
    cv2.rectangle(masked_frame, (center[0]-10+textX-5,center[1]-10-textY-25), (center[0]-10+textX+textsize[0]+5,center[1]-10-textY+textsize[1]-10), (255,255,255), -1)
    cv2.putText(masked_frame,label,(center[0]-10+textX,center[1]-10-textY),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)

    return masked_frame