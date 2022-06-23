import os
import csv
import glob
import random

from tensorflow import one_hot
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

import threading


class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


class DataSet():
    '''
    Management of UCF101 dataset
    '''

    # used (299, 299) or other used (224, 224)
    def __init__(self, seq_length_in_frames=30, class_limit=None, image_shape=(299, 299, 3), delta=5):
        '''
        `seq_length_in_frames` = length of input for LSTM
        `class_limit` = limit for classes (deafult all classes)
        `image_shape` = image dim for inputo for LSTM
        `delta` = step for each frame taken fom source
        '''
        self.seq_length_in_frames = seq_length_in_frames
        self.img_shape = image_shape
        self.class_limit = class_limit
        self.delta = delta

        self.data = self.get_data()
        self.classes = self.get_classes()
        self.data = self.filter_data()

    def get_data(self):
        '''load data from file data.csv'''
        with open(os.path.join('data', 'data.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        return data

    def get_classes(self):
        '''create class list'''
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])
        # sort class
        classes = sorted(classes)

        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def filter_data(self):
        '''filtering videos with less than set minimum number of frames (seq_length_in_frames) and aplaying class limitation'''
        filtered_data = []
        delta = 1 if self.delta == None else self.delta
        for item in self.data:
            if int(item[3]) >= (self.seq_length_in_frames * delta) and item[1] in self.classes:
                filtered_data.append(item)
        return filtered_data

    def get_train_test_split(self):
        '''returns train, test'''
        train, test = [], []
        for row in self.data:
            if row[0] == 'train':
                train.append(row)
            else:
                test.append(row)
        return train, test

    @threadsafe_generator
    def generator_sequence_and_return_X_Y(self, datatype: str, train_or_test: str, batch_size: int):
        '''Generator returning N features from the train/test set'''
        train, test = self.get_train_test_split()
        data = train if train_or_test == 'train' else test

        print(f'Generator z {len(data)} sampli ze zbioru: {train_or_test}')
        while True:
            X, Y = [], []
            for _ in range(batch_size):

                frames = None
                sample = random.choice(data)
                # Get the sequence from disk.
                frames = self.get_extracted_sequence(sample)
                # Chcek if loaded
                if frames is None:
                    raise ValueError("Nie znaleziono wygenerowanych cech, czy zostały już wygenerowane?")

                X.append(frames)
                class_id = self.get_class_id(sample[1])
                Y.append(class_id)
            yield (np.array(X), np.array(Y))

    def get_class_id(self, class_name: str):
        '''Returns the original index for a given class (str)'''
        # assign the id of the class name from the position in the list of classes and one hot it
        label_encoded = self.classes.index(class_name)
        label_hot = one_hot(label_encoded, len(self.classes))
        assert len(label_hot) == len(self.classes)

        return label_hot

    def get_extracted_sequence(self, sample):
        """returns one random saved features"""
        filename = sample[2]

        if self.delta == None:
            path = [os.path.join('data', 'futureSeq', 'Fast_forward', filename + '-' + 'futures' + str(self.seq_length_in_frames) + '.npy')]
        else:
            path = glob.glob(os.path.join('data', 'futureSeq', filename, 'futures_d' + str(self.delta) + '_s' + str(self.seq_length_in_frames), '*' + '.npy'))

        path = random.choice(path)
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_all_extracted_sequence(self, sample):
        """return all saved features"""
        filename = sample[2]

        if self.delta == None:
            path = [os.path.join('data', 'futureSeq', 'Fast_forward', filename + '-' + 'futures' + str(self.seq_length_in_frames) + '.npy')]
        else:
            path = glob.glob(os.path.join('data', 'futureSeq', filename, 'futures_d' + str(self.delta) + '_s' + str(self.seq_length_in_frames), '*' + '.npy'))
        seq = []
        for p in path:
            if os.path.isfile(p):
                seq.append(np.load(p))
        return seq

    @staticmethod
    def get_frames_from_sample(sample):
        """return all frames paths from sample"""
        return sorted(glob.glob(os.path.join('data', sample[0], sample[1], sample[2] + '_f*.jpg')))

    def calculate_data_len(self):
        suma = 0
        for vid in self.data:
            seq = self.get_all_extracted_sequence(vid)
            suma += len(seq)
        return suma

    @staticmethod
    def rescale_list_fast_forward(frames, size):
        """Returns every single video frame to match the number of frames in the size variable, 
        e.g. if you want a 5 frame recording from a 30 frame movie, you need to return every 6th frame"""
        assert len(frames) >= size

        delta = len(frames) // size

        return [frames[i] for i in range(0, len(frames), delta)][:size]

    @staticmethod
    def rescale_list_cut(frames, size=5, delta=5):
        """Returns frames every fixed time interval limited to a certain amount """
        assert len(frames) >= size

        return [frames[i] for i in range(0, len(frames), delta)][:size]

    def load_images_from_list(self, frame_list):
        '''Returns list of loaded images from frames path list'''
        return [self.procces_image(img, self.img_shape) for img in frame_list]

    def procces_image(img, image_shape):
        h, w, _ = image_shape
        image = load_img(img, target_size=(h, w))

        # Turn it into numpy, normalize and return.
        img_arr = img_to_array(image)
        x = (img_arr / 255.).astype(np.float32)
        return x


def main():
    #exp:
    data = DataSet(seq_length_in_frames=5, delta=5)
    print(data.classes)


if __name__ == '__main__':
    main()
