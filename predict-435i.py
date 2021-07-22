import os
import sys
import cv2
import yaml

from ctypes import *
import random
import time

import darknet
import argparse





import numpy as np
import pyrealsense2 as rs

YAML_FILE = "bs-rover.yml"
INPUT = "0"
OUTPUT = "output.mp4"

SAVE_VIDEO = False

class YoloCfg():
    def __init__(self,yml):
        self.input = INPUT
        self.weights = yml["weights"]
        self.config_file = yml["config_file"]
        self.data_file = yml["data_file"]
        self.thresh = 0.80


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

def calc_average(image, bbox):
    x, y, w, h = bbox
    w,h = w//5,h//5
    roi = image[y-h:y+h,x-w:x+w]
    return np.sum(roi)/roi.size



# def start_plot():
#     while True:
#         pass


if __name__ == '__main__':
    # args = parser()
    f = open(YAML_FILE)
    yml = yaml.load(f,Loader=yaml.FullLoader)
    f.close()

    # init YOLO
    args = YoloCfg(yml)

    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)


    # init d435i
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        sys.exit()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        vw,vh = 1280, 720
        video_color = cv2.VideoWriter("_color_out.mp4", fourcc, fps, (vw,vh))
        video_depth = cv2.VideoWriter("_depth_out.mp4", fourcc, fps, (vw,vh))


    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    
    # 实际距离为depth_image的值*depth_scale
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    depth_sensor.set_option(rs.option.emitter_enabled, 0)

    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    print(depth_intrinsics)

    n = 0
    fileName = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fo = open(f"txtdata/{fileName}.txt","w")
    fp = open(f"txtdata/{fileName}-ss.txt","w")
    start_time = time.time()

    while True:
        bb = None
        depth_point = None
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        detections = darknet.detect_image(network, class_names, img_for_detect, thresh=args.thresh)
        darknet.free_image(img_for_detect)

        
        detections_adjusted = []
        now = time.time() - start_time
        for label, confidence, bbox in detections:
            bbox_adjusted = convert2original(frame, bbox)
            detections_adjusted.append((str(label), confidence, bbox_adjusted))
            
            ave_depth = calc_average(depth_image,bbox_adjusted)*depth_scale

            depth_pixel = bbox_adjusted[:2]
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, depth_pixel, ave_depth)
            # print(bbox_adjusted,ave_depth,depth_point)
            # fo.write("{:.2f} {},{},{},{} {},{},{}\r".format(now,*bbox_adjusted,*depth_point))
            bb = bbox_adjusted
        
        if SAVE_VIDEO:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            video_color.write(frame)
            video_depth.write(depth_colormap)





        image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        cv2.imshow('Inference', image)
    
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        # if key == ord(" "):
        if bb != None and depth_point != None:
            n += 1
            print("{:.2f}".format(now),depth_point)
            # print("{:.2f}".format(now),ave_depth,aligned_depth_frame.get_distance(*bb[:2]))
            # print("{:.2f}".format(now),ave_depth,aligned_depth_frame.get_distance(*bb[:2]))
            # print(now,ave_depth,aligned_depth_frame.get_distance(*bb[2:]))
            # print(now,bb,depth_point,)
            fp.write("{:.2f} {},{},{},{} {},{},{}\r".format(now,*bb,*depth_point))
            # np.save("npydata/i-{:05d}.npy".format(n),frame)
            # np.save("npydata/d-{:05d}.npy".format(n),depth_image) 
            # print(f"{n} save done")


    if SAVE_VIDEO:
        video_color.release()
        video_depth.release()



