from flask import Flask, request
from logging import Logger
from web.converters import base64_to_opencv_img
from train_cnn import pre_process_fn #todo refactor this into a seperate file rather than a script.
app = Flask(__name__)


"""
WE input to the route a JSON object (as post)/

{
    "base64": "<BASE 64 STRING OF IMAGE GOES HERE...>"
}
"""



@app.route('/getLiveness', methods=['POST'])
def get_cnn_output():
    base64_to_opencv_img = request.data['base64']
    # First, base64 to opencv image
    img = base64_to_opencv_img(base64_img)
    preprocessed_img = pre_process_fn(img)

    # First, insert code here to calculate the quality metric.

    # TODO insert prediction here for the quality metricx


    # The return the output prediction of the LDA.
    return output
