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

data = DataSet(30)


def build_feature_model():
    base_model = InceptionV3(weights='imagenet', include_top=True)

    return Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


model = build_feature_model()


def main():
    pbar = tqdm(total=len(data.data))
    for movie in data.data:
        path = os.path.join('data', 'futureSeq', movie[2].replace('_f0000', '') + '-futures' + str(data.seq_length_in_frames))

        if os.path.isfile(path + '.npy'):
            pbar.update(1)
            continue

        frames = []
        frames_path = data.get_frames_from_sample(movie)
        frames_path = data.rescale_list_fast_forward(frames_path, data.seq_length_in_frames)
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

        print('Zapisywanie sekwencji: ', path)
        pbar.update(1)
        np.save(path, frames)


if __name__ == '__main__':
    main()
