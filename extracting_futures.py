"""
extract features from images to hard drive with given param. (fast forward method)
"""
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
import numpy as np
from tqdm import tqdm

from data import DataSet

SEQ_LENGHT = 25
DELTA = 2


def build_feature_model():
    base_model = InceptionV3(weights='imagenet', include_top=True)

    return Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


data = DataSet(SEQ_LENGHT, delta=DELTA)
model = build_feature_model()  # InceptionV3
pbar = tqdm(total=len(data.data))
for movie in data.data:
    frames = []
    frames_path = data.get_frames_from_sample(movie)
    frames_path = frames_path[::DELTA]
    movie_name = movie[2].replace('_f0000', '')
    count = 0
    for frame_path in frames_path:
        # print(frame_path)
        h, w, _ = data.img_shape
        img = image.load_img(frame_path, target_size=(h, w))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features = np.squeeze(features, axis=0)
        frames.append(features)
        if len(frames) == SEQ_LENGHT:
            folder_path = os.path.join('data', 'futureSeq', movie_name, 'futures_d' + str(DELTA) + '_s' + str(data.seq_length_in_frames))
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            path_dest = os.path.join('data', 'futureSeq', movie_name, 'futures_d' + str(DELTA) + '_s' + str(data.seq_length_in_frames), str(count))
            print('Saving seq: ', path_dest)
            np.save(path_dest, frames)
            count += 1
            frames = []
    pbar.update(1)
