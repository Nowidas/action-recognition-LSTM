"""
Script for moving sequences
"""
import glob
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
import numpy as np
from tqdm import tqdm
import shutil

from data import DataSet

data = DataSet(5)
DELTA = 5

for movie in data.data:
    filename = movie[2]

    if not os.path.isdir(os.path.join(os.path.join('data', 'futureSeq', movie[2].replace('_f0000', '')))):
        os.mkdir(os.path.join(os.path.join('data', 'futureSeq', movie[2].replace('_f0000', ''))))

    paths = glob.glob(os.path.join(os.path.join('data', 'futureSeq', movie[2].replace('_f0000', '') + '-futures1s' + str(data.seq_length_in_frames), '_*.npy')))
    # print(paths)
    for it, path in enumerate(paths):
        pathDest = os.path.join('data', 'futureSeq', movie[2].replace('_f0000', ''), 'futures_d' + str(DELTA) + '_s' + str(data.seq_length_in_frames), str(it) + '.npy')

        # print(path, ' -> ', pathDest)
        if not os.path.isdir(os.path.join('data', 'futureSeq', movie[2].replace('_f0000', ''), 'futures_d' + str(DELTA) + '_s' + str(data.seq_length_in_frames))):
            os.mkdir(os.path.join('data', 'futureSeq', movie[2].replace('_f0000', ''), 'futures_d' + str(DELTA) + '_s' + str(data.seq_length_in_frames)))

        os.rename(path, pathDest)
