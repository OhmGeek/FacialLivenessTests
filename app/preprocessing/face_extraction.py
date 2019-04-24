import cv2
import face_recognition
import numpy as np


def get_largest_bounding_box(locations):
    if len(locations) == 0:
        return None
    w = max(locations, key=lambda x: np.linalg.norm(x[0] - x[2]) * np.linalg.norm(x[1] - x[3]))
    return w


def pre_process_fn(image_arr):
    original_shape = image_arr.shape
    image_arr = image_arr.astype(np.uint8)
    locations = face_recognition.face_locations(image_arr, number_of_times_to_upsample=0, model='cnn')

    max_loc = get_largest_bounding_box(locations)
    # If there's an error, just use the whole image.
    if max_loc is None:
        return cv2.resize(image_arr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    # Otherwise, isolate the face.
    top, right, bottom, left = max_loc

    dist = max(abs(bottom - top), abs(right - left))

    new_bottom = top + dist
    new_right = left + dist
    face_image = image_arr[top:new_bottom, left:new_right]

    # Now, to fix a bug in Keras, resize this image.
    face_image = cv2.resize(face_image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    return (face_image)


def preprocess_fn_all(x):
    output = []
    for img in x:
        output.append(pre_process_fn(img))
    output = np.array(output)

    return output