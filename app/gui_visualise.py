#####################################################################

# Example : load and display a set of images from a directory
# basic illustrative python script

# For use with provided test / training datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
from datasets.nuaa import NUAADataset
from datasets.replayattack import ReplayAttackDataset
from logging import Logger

dataset = NUAADataset(Logger("nuaa"), "/home/ryan/datasets/nuaa/")
dataset.pre_process()
imgs = dataset.get_all_datasets()

# display all images in directory (sorted by filename)

for img_num in range(0, len(imgs)):
    img = imgs[img_num]
    cv2.imshow('the image',img)
    key = cv2.waitKey(200) # wait 200ms
    if (key == ord('x')):
        break


# close all windows

cv2.destroyAllWindows()

#####################################################################