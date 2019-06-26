# LivenessTests
Liveness Tests For Facial Recognition. 4th year project for M.Eng Computer Science at Durham University. Supervised by Andrei Krokhin.

## Introduction
The aim of this project is to understand the realm of facial liveness tests, and produce an investigation into the ease of creating a liveness as a service platform.

This project creates three new liveness tests, one for whole image assessment, one for facial structure assessment, and one for detecting mask attacks.


## Dependencies:
- Python 3.5+
- Pip
- Pipenv for dependency management
- Numpy
- OpenCV
- SkImage
- Pywavelets
- Sci-kit Learn
- H5py
- Keras
- Tensorflow (either tensorflow for CPU, or tensorflow-GPU for GPU)
- face-recognition
- joblib
- pyvideoquality, v0.1.0-beta.5", from git = "https://github.com/OhmGeek/video-quality.git" (adapted to a Python library by me).

## Folder Structure
### app/
This contains all the source code necessary to run the program. README.md in the app folder explains what all the scripts do, and goes into detail about the
namespaces.

### datasets/
This needs to be a blank folder with a specific folder structure. Why? It's where the cached datasets are stored (in h5 format). This is done to improve performance.


Under the datasets folder, 3 folders are needed. The files shown below might not exist, but they will be created when running the application.

```
datasets/
    mad/
        mad.h5
    nuaa/
        nuaa.h5
    replay-attack/
        replayAttackdevel.h5
        replayAttacktest.h5
```

### models/
This is where functional models are kept. CNN and VoxNet used h5 to store weights, while the WIQA method uses pickle (because it's simple and easy).

## How to run?
Navigate to the app folder.

Install pipenv and Python3.7 then run:

```
pipenv install
pipenv shell
```

Then run the python script as normal. A virtualenv with tensorflow CPU will be created for use.
