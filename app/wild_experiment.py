from keras.models import load_model
from vrn import custom_layers
import visvis as vv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from liveness.vox.reconstruction.vrn import FaceVoxelBuilder
import logging
# custom_objects = {
#     'Conv': custom_layers.Conv,
#     'BatchNorm': custom_layers.BatchNorm,
#     'UpSamplingBilinear': custom_layers.UpSamplingBilinear
# }
# model = load_model('vrn-unguided-keras.h5', custom_objects=custom_objects)
builder = FaceVoxelBuilder(logging.Logger(""))


img = cv2.imread('/home/ryan/datasets/nuaa/ClientRaw/0001/0001_00_00_02_2.jpg')
volRGB = builder.build_3d(img)
# img = cv2.resize(img, (192, 192))
# b,g,r = cv2.split(img)
# img = cv2.merge([r,g,b])
# img = np.swapaxes(img, 2, 0)
# img = np.swapaxes(img, 2, 1)
# img = np.array([img])

# pred = model.predict(img)
# vol = pred[0] * 255
# im = img[0]
# im = np.swapaxes(im, 0, 1)
# im = np.swapaxes(im, 1, 2)

# volRGB = np.stack(((vol > 1) * im[:,:,0],
#                    (vol > 1) * im[:,:,1],
#                    (vol > 1) * im[:,:,2]), axis=3)
for i, p in enumerate(volRGB[80:180:5]):
    plt.subplot(5, 4, i + 1)
    plt.imshow(p)


vv.clf()

v = vv.volshow(volRGB, renderStyle='iso')

l0 = vv.gca()
l0.light0.ambient = 0.9 # 0.2 is default for light 0
l0.light0.diffuse = 1.0 # 1.0 is default

a = vv.gca()
a.camera.fov = 0 # orthographic

vv.use().Run()