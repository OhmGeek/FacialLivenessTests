"""
This file uses the VRN neural network to convert a 2D image into a 3D voxel structure.

This is then passed for classification.
"""
def preprocess_image_for_builder(image, new_size=(192,192)):
    # This takes an image, preprocesses the image.
    ## First, resize to new size (default: 192 x 192)
    image = cv2.resize(image, new_size)

    ## B,G,R -> R,G,B
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    image = np.swapaxes(image, 2, 0)
    image = np.swapaxes(image, 2, 1)

    # R,G,B -> image
    image = np.array([image])

    return image

class FaceVoxelBuilder(object):
    def __init__(self, logger):
        self._logger = logger

    def build_3d(self, image):
        img = preprocess_image_for_builder(image)