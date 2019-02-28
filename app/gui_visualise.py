
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
model_outputs = str(sys.argv[2]) # 1 => use Certainty, 2=> real, fake

dataset = NUAADataset(Logger("nuaa"), "/home/ryan/datasets/nuaa/")
dataset.pre_process()
print(dataset)
imgs = dataset.read_dataset("ImposterRaw")
model = ModelReader().read_from_file(model_path)
objects = None

# Setup the labels
if(model_outputs == "1"):
    objects = ('Certainty', )
else:
    objects = ('Real', 'Fake')

for img in imgs:
    print(img.shape)
    print(model.__dict__)
    try:
        prediction = model.evaluate(np.array(img))

    except Exception as ex:
        print("Skipping image because of exception", ex)
        continue
    print(prediction)
    # Create a figure
    fig = plt.figure()
    fig.add_subplot(111)

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Do the plot
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
    key = cv2.waitKey(1) # wait 200ms
    if (key == ord('x')):
        break

    exit()

# close all windows

cv2.destroyAllWindows()

#####################################################################
