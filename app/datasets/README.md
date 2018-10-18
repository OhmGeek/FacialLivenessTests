# app.datasets
This section deals with the datasets that we use to test/train.

Each dataset can contain two types of content: images (which is just one frame), and video.

With video, we need further configuration to process the dataset: do we use each individual frame as an image,
or every second? This is all customisable via a config file.

We also need to divide into two seperate sets: training set for training the mode, and test set for testing the system.


## Data storage:
FIles are going to be large. Therefore, we can't necessarily load everything into memory at once.
To save this, we can use h5py (and by extension, the hdf5 file structure) to use hard drive memory.
Keras has built in support, and so does numpy, so we can use this as a dataset standard.

We first need a migration script for each dataset - we need to load each image as a dataset into the h5py memory structure as an array
(and with video, taking into account the necessary settings).

Then, we save the dataset file somewhere to then use in the final stage. The code below is example code
for writing the dataset to a file:

```python
import numpy as np
import h5py

data_to_write = '<LOAD DATASET HERE>'

with h5py.File('name-of-file.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=data_to_write)
```

Reading the dataset from the file can then be done using:
```python
data = None
with h5py.File('name-of-file.h5', 'r') as hf:
    data = hf['name-of-dataset'][:]

if(data == None):
    raise Exception("Dataset not loaded correctly")
else:
    pass #do something...
```