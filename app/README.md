# Liveness Testing for the Web!
## Introduction
This system is designed to train and test three new models, hopefully some of which can be deployed into
a LivenessTest-As-A-Service system.

### Metric 1: Whole Image Quality Assessment
Based on the Image Quality Assessment method, instead of isolating the face and running the code on this, instead
analyse the quality of the WHOLE image.

### Metric 2: CNN based Liveness Test
This method utilises pretrained ImageNet models. As ImageNet classification and liveness are fairly similar problems,
coupled with the availability and reliability of existing ImageNet classifiers on the global web, the aim was to adapt
this type of model to the problem of facial liveness.

### Metric 3: VoxNet based Liveness Test for detecting Mask Attacks
And now for something completely different...

This method uses an existing 2D to 3D face reconstruction method to create a voxel representation of a person's face.

Then, this is fed through a VoxNet based classifier.

This didn't perform that well, as the reconstruction time was very high (over 5 seconds), coupled with the poor performance of 
VoxNet due to high memory consumption. This method was scrapped, but the code still exists in case anyone wants to adapt it/
improve on this idea.

## Scripts
The top level folder contains a set of scripts for training, testing, debugging and data collection purposes.

### get_quality_vector.py -- WIQA Test -- Debugging
This file is a debugging tool for the Whole Image Quality Assessment method. Given an image (hard coded),
it calculates the specified metrics, and prints the metric vector.

### graph_showing_test_predictions.py -- All -- Data Collection
This file fetches predictions from both the WIQA and the ResNet based liveness test, and produces a graph
showing predictions for both. This is useful to determine the linearly seperable nature of the problem. It was thought
that the outputs from both could be fed into a perceptron, which can be used as a consolidation layer to the overall system.
This was designed to test that hypothesis, which was partly true, but hard to justify from the graph produced.

### measure_time_for_predict.py -- All -- Data Collection
This file creates predictions from both models, discarding the overall outcome. The main focus was to time the loading time
for each model (using wall clock time). For each prediction, we take the image individually and time the prediction time
(including preprocessing). In total, the prediction time is averaged over the 100 images (or whatever is selected).

This happens for both, and produced the time results in the paper.

### preprocess_voxnet.py -- VoxNet -- Preprocessing
This file is sourced from the original VoxNet source, designed to preprocess the SUOD dataset. The initial aim was to
create a pretrained VoxNet model, which would then be adapted. However, this didn't work in practice.
The code still exists here.

### test_cnn.py -- CNN -- Testing
This file tests the CNN model. Given a pretrained CNN model (which needs to be specified as a h5 file in the code),
and a specific test set, the data can be loaded from the test set and test results produced (including the confusion
matrix).

### test_quality.py -- WIQA Test -- Testing
This file tests the quality metric. It loads a pretrained classifier (an LDA), and then tests the accuracy of the outcome, along
with creation of a confusion matrix.

### train_voxnet.py -- VoxNet -- Training
This is the code to train the VoxNet model. One might need to adjust the voxnet model to improve filters per conv layer if you have any
hope of training accurately. The number of filters was drastically reduced due to memory issues. This code uses the custom VoxNEt generator.

### train_cnn.py -- CNN -- Training
This file trains the CNN model, then saves the result, then tests the result and produces similar output to the test file (though
it might be slightly different in terms of accuracy due to the batch nature of testing conducted). 

### train_quality.py -- WIQA Test -- Training
This trains the W-IQA classifier based on an image set. The model is saved after training, and testing commences too, yielding
similar results to the test script.

### reconstruct_face_2d_to_3d.py -- VoxNet -- Debug
This is an example demo of 2D to 3D reconstruction. This only contains slices, but further code can be found online using the
vv tool to produce a 3D model. This code also measures the time taken to reconstruct, which was used in the paper.


## Modules
### app.datasets
This namespace contains anything to do with datasets/data, including custom generators for training, generic dataset implementation
and the implementations of Mask Attack Dataset, NUAA Dataset, and ReplayAttack Dataset.

#### app.datasets.generators
Contain the generators which were used.

image_data_generator.py was used for standard 2D images (to allow face cropping on the fly).

voxnet_data_generator.py was used for 2D to 3D conversion (to allow on the fly 3D reconstruction).


### app.liveness
All the code relating to the 3 different liveness tests. 

#### app.liveness.cnn
CNN test code

residual/* contains the code for the residual network model.

alexnet.py contains the function to generate a basic AlexNet model to be used.

#### app.liveness.quality
Quality Metric code.

classifiers/* contains the models for both the SVM and LDA models. The LDA model was the one assessed, while the code for the SVM model
exists for the record.

helpers/* contains code for calculating the Blind Image Quality Index, with biqi_data being used as a folder for the file system interaction.

metrics/* contain the classes for each metric. There is a generic file which is the Abstract implementation of two types of metrics,
and each metric extends one of these two metrics.

The metric_vector.py file creates an abstract class for generating a vector, and also creates the reference implementation for
calculating the metric vector. The parallelisation is done using Joblib. 

The preprocessing.py file contains the code needed for the preprocessing of data (before being fed into the classifier). This contains
the main code that actually generates the metrics and metric vectors.

#### app.liveness.vox
VoxNet code.

classification/* contains the code for VoxNet.

reconstruction/* contains the code for reconstructing using VRN and the custom layers that are required.

### app.preprocessing
This contains code for preprocessing the dataset (not in the models, but before being fed into the models).

The face_extraction.py file contains the code used by the CNN method to extract a person's face from teh input image,
and resizing it appropriately.

The suod_helper.py is used by the VoxNet code in the SUOD helper. It comes from someone else's implementation, not my own.



