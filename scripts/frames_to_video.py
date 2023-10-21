import cv2
import argparse
import os

def frames_to_video(input_dir, output_dir, fps):
    frames_path_dir = os.listdir(input_dir)
    frames_path_dir.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # get all images in the input directory
    img_array = []
    for frame in frames_path_dir:
        filename = os.path.join(input_dir, frame)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    input_name = input_dir.split('/')[-1]
    # create the video writer
    out = cv2.VideoWriter(os.path.join(output_dir, input_name + '_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    # write the video
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data/', help='directory with images')
    parser.add_argument('--output_dir', type=str, default='./', help='directory to save the video')
    parser.add_argument('--fps', type=int, default=5, help='frames per second')
    args = parser.parse_args()

    frames_to_video(args.input_dir, args.output_dir, args.fps)