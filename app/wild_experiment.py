"""
3D reconstruction of faces from 2D images: a test.
"""
import visvis as vv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from liveness.vox.reconstruction.vrn import FaceVoxelBuilder
import logging

builder = FaceVoxelBuilder(logging.Logger(""))

# load in image
img = cv2.imread('/home/ryan/datasets/nuaa/ClientRaw/0001/0001_00_00_02_2.jpg')
imgs = []
imgs.append(img)
imgs.append(img)
imgs = np.array(imgs)
# build 3D
volRGB = builder.build_3d_multiple(imgs)
print(volRGB.shape)
# Render slices on screen
for i, p in enumerate(volRGB[80:180:5]):
    plt.subplot(5, 4, i + 1)
    plt.imshow(p)

plt.show()
