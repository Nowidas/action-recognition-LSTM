"""
Train LSTM for difrent parm.
"""

import csv
from traceback import print_tb
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from models import Model
from data import DataSet
import time
import os.path
import tensorflow as tf
import numpy as np
import keras.backend as K


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def train(data_type, seq_length, model, dropout, recurent_dropout, saved_model=None, class_limit=None, image_shape=None, batch_size=32, nb_epoch=100, initial_epoch=0, RUN_TEST='', delta=5):

    # helper time
    time_callback = TimeHistory()

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', 'exp2', model + RUN_TEST + '-' + str(dropout) + '-' + str(recurent_dropout) + '-' + str(timestamp) + '.log'))

    # Get the data and process it.
    data = DataSet(seq_length_in_frames=seq_length, class_limit=class_limit, image_shape=image_shape, delta=delta)

    # Calculate steps_per_epoch
    # steps_per_epoch = (len(data.data) * 0.7) // batch_size
    steps_per_epoch = (18729) // batch_size  # 5_20

    # Get generators.
    generator = data.generator_sequence_and_return_X_Y(data_type, 'train', batch_size)
    val_generator = data.generator_sequence_and_return_X_Y(data_type, 'test', batch_size)

    # Get the model.
    rm = Model(len(data.classes), model, seq_length, saved_model, dropout=dropout, recurent_dropout=recurent_dropout)

    # Fit!
    hist = rm.model.fit(x=generator, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, verbose=1, callbacks=[early_stopper, csv_logger, time_callback], validation_data=val_generator, validation_steps=30, workers=4, initial_epoch=initial_epoch)

    index_of_max = np.argmax(hist.history['val_accuracy'])
    with open(os.path.join('data', 'logs', 'exp2', model + '_acc' + '.csv'), 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow([model + RUN_TEST + '-' + str(round(dropout, 1)) + '-' + str(round(recurent_dropout, 1))] + [hist.history['val_accuracy'][index_of_max]] + [hist.history['val_loss'][index_of_max]] + [hist.history['val_top_k_categorical_accuracy'][index_of_max]])

    with open(os.path.join('data', 'logs', 'exp2', model + '_time' + '.csv'), 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow([model + RUN_TEST + '-' + str(round(dropout, 1)) + '-' + str(round(recurent_dropout, 1))] + time_callback.times)


def main():
    model = 'lstm'
    RUN_TEST = ''
    saved_model = None
    class_limit = None
    seq_length = 20
    load_to_memory = False
    batch_size = 32
    nb_epoch = 1000
    delta = 5

    if model in ['lstm', 'BiLSTM', 'ResLSTM', 'DenseLSTM', 'Dense_Bi_LSTM']:
        data_type = 'future_seq'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    for dropout in np.arange(0, 0.8 + 0.2, 0.2):
        for recurent_dropout in np.arange(0, 0.8 + 0.2, 0.2):
            print('Training ', model, ': ', dropout, ' ', recurent_dropout)
            train(data_type, seq_length, model, saved_model=saved_model, class_limit=class_limit, image_shape=image_shape, batch_size=batch_size, nb_epoch=nb_epoch, RUN_TEST=RUN_TEST, dropout=dropout, recurent_dropout=recurent_dropout, delta=delta)


if __name__ == '__main__':
    main()
