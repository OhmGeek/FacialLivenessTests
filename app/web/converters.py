import cv2
import numpy as np

def base64_to_opencv_img(base64_code):
    """Convert a Base64 encoded image to an opencv image.
    
    Arguments:
        base64_code {string} -- A base64 encoded image, as string
    
    Returns:
        OpenCV Image -- An opencv image.
    """

    nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

