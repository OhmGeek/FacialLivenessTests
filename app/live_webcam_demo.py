#####################################################################

# Example : load and display a set of images from a directory
# basic illustrative python script

# For use with provided test / training datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import numpy as np
import cv2
from liveness.cnn.residual.model import ResidualNetwork
from liveness.quality.classifiers.lda import QualityLDAModel
from preprocessing.face_extraction import preprocess_fn_all, pre_process_fn
import logging

logger = logging.Logger('model')
# Create the models first.
model_cnn = ResidualNetwork(logger)
model_cnn.load('/home/ryan/Documents/dev/LivenessTests/models/cnn_v3.h5')

model_wiqa = QualityLDAModel(logger)
model_wiqa.load('/home/ryan/Documents/dev/LivenessTests/models/lda_model_v2.pkl')

# cap = cv2.VideoCapture(0)
# Video for fake:
# cap = cv2.VideoCapture('/home/ryan/datasets/replayAttackDB/replayattack-test/test/attack/fixed/attack_mobile_client031_session01_mobile_video_controlled.mov')

# Video for real:
cap = cv2.VideoCapture('/home/ryan/datasets/replayAttackDB/replayattack-test/test/real/client028_session01_webcam_authenticate_controlled_2.mov')
labels = ['fake', 'real']

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    preprocessed_frame = pre_process_fn(frame)
    output = model_cnn.evaluate(np.array([preprocessed_frame]))
    classification_cnn = labels[np.argmax(output)]

    # print("CNN output: ", classification_cnn)
    cnn_out_str = "ResNet Output: %s" % classification_cnn

    # Now do the same with the other model.
    ## TODO resize the frame to the size of Replayattack images (because i bet this is the problem, as we have strange aspect ratio).
    # frame_resized = cv2.resize(frame, (320, 240))
    output = int(model_wiqa.evaluate(np.array([frame]))[0])
    classification_wiqa = labels[output]
    wiqa_out_str = "WIQA Output: %s" % classification_wiqa

    cv2.putText(frame, wiqa_out_str, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.putText(frame, cnn_out_str, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('face region', preprocessed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
