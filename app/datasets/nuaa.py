from datasets.generic import Dataset
import numpy as np
import cv2
import glob

class NUAADataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        super().__init__()

    def pre_process(self):
        # First, load imposter images.

        imposter_images = []

        for img_filename in glob.iglob(self.filename + 'ImposterRaw/**/*.jpg'):
            img = cv2.imread(img_filename, mode='RGB')
            imposter_images.append(img)


        print(imposter_images)
    

