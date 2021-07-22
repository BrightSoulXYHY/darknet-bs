#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-07-20 10:13:59
# @Author  : BrightSoul (653538096@qq.com)

import os
import shutil


# img_paths = [
#     "F:/学习/2021-05ICRA/树莓派小车车/realsense/realsense-rover-A",
# ]

img_paths = "F:/学习/2021-05ICRA/树莓派小车车/realsense/realsense-rover"
lable_path = "F:/学习/2021-05ICRA/树莓派小车车/realsense/realsense-rover-yolo"
extension = ".jpg"

L = os.listdir(lable_path)
L.remove("classes.txt")

for i in L:
    base_name = os.path.splitext(os.path.basename(i))[0]+extension
    source = os.path.join(img_paths, base_name)
    target = os.path.join(lable_path, base_name)
    shutil.copyfile(source, target)

print("copy done")