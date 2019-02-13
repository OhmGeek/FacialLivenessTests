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
import sys
from datasets.nuaa import NUAADataset
from datasets.replayattack import ReplayAttackDataset
from logging import Logger
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from liveness.io.reader import ModelReader

model_path = str(sys.argv[1])

dataset = ReplayAttackDataset(Logger("nuaa"), "/home/ryan/datasets/ReplayAttackDB/")
dataset.pre_process()
print(dataset)
imgs = dataset.read_dataset("attack")
model = ModelReader().read_from_file(model_path)
# display all images in directory (sorted by filename)
for img in imgs:
    print(img.shape)
    prediction = model.evaluate(img)
    print(prediction)
    # Create a figure
    fig = plt.figure()
    fig.add_subplot(111)

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Do the plot
    objects = ('Real', 'Fake')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, prediction, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Certainty')
    plt.title('Model prediction of certainty')
    
    # Now take the output bar chart as numpy array image.
    bar_chart = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    bar_chart = bar_chart.reshape(fig.canvas.get_width_height()[::-1] + (3,))


    output = np.hstack((img, bar_chart))
    cv2.imshow('Prediction',img)
    key = cv2.waitKey(200) # wait 200ms
    if (key == ord('x')):
        break


# close all windows

cv2.destroyAllWindows()

#####################################################################