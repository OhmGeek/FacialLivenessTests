import cv2


class ImageBlurrer:
    def __init__(self, sigma=0, kernel=(5, 5)):
        """
        Create the image blurrer component
        :param sigma: The sigma value to use. Defaults to 0 (which is automatic mode)
        :param kernel: Kernel size to use. Default is (5,5)
        """
        self.sigma = sigma
        self.kernel = kernel

    def blur(self, image):
        return cv2.GaussianBlur(image, self.kernel, self.sigma)
