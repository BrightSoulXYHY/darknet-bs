import os
import glob
import yaml

import cv2
from ctypes import *
import random
import time
import darknet
import argparse

YAML_FILE = "bs-rover.yml"
# INPUT = "img/i-00003.png"
INPUT = "img/00985.jpg"

class YoloCfg():
    def __init__(self,yml):
        self.input = INPUT
        self.weights = yml["weights"]
        self.config_file = yml["config_file"]
        self.data_file = yml["data_file"]
        self.thresh = 0.25


# def parser():
#     parser = argparse.ArgumentParser(description="YOLO Object Detection")
#     parser.add_argument("--input", type=str, default="map6-h7-y45_00179.jpg",
#                         help="video source. If empty, uses webcam 0 stream")
#     parser.add_argument("--weights", default="./bs-config/bs-drone_6000.weights",
#                         help="yolo weights path")
#     parser.add_argument("--config_file", default="./bs-config/bs-drone.cfg",
#                         help="path to config file")
#     parser.add_argument("--data_file", default="./bs-config/bs-drone.data",
#                         help="path to data file")
#     parser.add_argument("--thresh", type=float, default=.25,
#                         help="remove detections with lower confidence")
#     return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))



if __name__ == '__main__':
    # args = parser()
    f = open(YAML_FILE)
    yml = yaml.load(f,Loader=yaml.FullLoader)
    f.close()


    args = YoloCfg(yml)
    check_arguments_errors(args)
    
    images = load_images(args.input)
    
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    
    
    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")
        
        frame = cv2.imread(image_name)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        detections = darknet.detect_image(network, class_names, img_for_detect, thresh=args.thresh)

        darknet.free_image(img_for_detect)

        detections_adjusted = []
        for label, confidence, bbox in detections:
            bbox_adjusted = convert2original(frame, bbox)
            detections_adjusted.append((str(label), confidence, bbox_adjusted))
            
            # fo.write("{} {},{},{},{}\r".format(n,*bbox_adjusted))

        frame = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        print(detections_adjusted)
        if len(images) == 1:
            cv2.imshow('Inference', frame)
            cv2.waitKey()



