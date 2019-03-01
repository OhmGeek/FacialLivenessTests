
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

dataset = ReplayAttackDataset(Logger("nuaa"), "/home/ryan/datasets/replayAttackDB/")
dataset.pre_process()
print(dataset)
imgs = dataset.read_dataset("attack")
model = ModelReader().read_from_file(model_path)
objects = None

# Setup the labels
if(model_outputs == "1"):
    objects = ('Certainty', )
else:
    objects = ('Real', 'Fake')

counter = 0
for img in imgs:
    plt.close()
    counter += 1
    print("###### IMAGE ", counter)
    try:
        prediction = model.evaluate(np.array([img]))

    except Exception as ex:
        print("Skipping image because of exception", ex)
        continue
    
    print(prediction)
    # Create a figure
    fig = plt.figure()
    fig.add_subplot(111)

    print("#############################################")
 
    # Do the plot
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, prediction, align='center')
    plt.xticks(y_pos, objects)
    plt.ylabel('Certainty')
    plt.title('Model prediction of certainty')
     # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()
    # Now take the output bar chart as numpy array image.
    bar_chart = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    bar_chart = bar_chart.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_img = cv2.resize(img, dsize=(bar_chart.shape[1], bar_chart.shape[0]), interpolation=cv2.INTER_CUBIC)
    output = np.hstack((output_img, bar_chart))
    cv2.imshow('Prediction',output)
    key = cv2.waitKey(1000 ) # wait 200ms
    if (key == ord('x')):
        break


# close all windows

cv2.destroyAllWindows()

#####################################################################
