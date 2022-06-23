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
from keras.utils.layer_utils import count_params


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def train(data_type, seq_length, model, units, depth, seq_or_last, saved_model=None, class_limit=None, batch_size=32, nb_epoch=100, initial_epoch=0, RUN_TEST='', delta=5):
    # Helper: Save the model.
    # checkpointer = ModelCheckpoint(
    # filepath=os.path.join('data', 'checkpoints', model + '_' + RUN_TEST + '_d' + delta + '_s' + seq_length +
    #                       '.{epoch:03d}-{val_loss:.3f}.hdf5'),
    #     verbose=1,
    #     save_best_only=True)

    # helper time
    time_callback = TimeHistory()

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('trash', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    t = 'seq' if seq_or_last else 'last'
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', 'exp1', model + RUN_TEST + '-' + t + '-' + str(depth) + '-' + str(units) + '-' + str(timestamp) + '.log'))

    # Get the data and process it.
    data = DataSet(seq_length_in_frames=seq_length, class_limit=class_limit, delta=delta)

    # Get samples per epoch.
    # steps_per_epoch = (len(data.data) * 0.7) // batch_size
    steps_per_epoch = (18729) // batch_size  # 5_20

    # Get generators.
    generator = data.generator_sequence_and_return_X_Y(data_type, 'train', batch_size)
    val_generator = data.generator_sequence_and_return_X_Y(data_type, 'test', batch_size)

    # Get the model.
    rm = Model(len(data.classes), model, seq_length, saved_model, units=units, depth=depth, seq_or_last=seq_or_last)

    # Plot model
    # tf.keras.utils.plot_model(
    #     rm.model, model + RUN_TEST + '.png', show_shapes=True, dpi=300)

    trainable_count = count_params(rm.model.trainable_weights)
    with open(os.path.join('data', 'logs', 'exp1', model + '_param' + '.csv'), 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow([model + RUN_TEST + '-' + t + '-' + str(depth) + '-' + str(units)] + [trainable_count])

    # Fit!
    # Use fit generator.
    hist = rm.model.fit(x=generator, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, verbose=1, callbacks=[tb, early_stopper, csv_logger, time_callback], validation_data=val_generator, validation_steps=30, workers=4, initial_epoch=initial_epoch)

    index_of_max = np.argmax(hist.history['val_accuracy'])
    with open(os.path.join('data', 'logs', 'exp1', model + '_acc' + '.csv'), 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow([model + RUN_TEST + '-' + t + '-' + str(depth) + '-' + str(units)] + [hist.history['val_accuracy'][index_of_max]] + [hist.history['val_loss'][index_of_max]] + [hist.history['val_top_k_categorical_accuracy'][index_of_max]])

    with open(os.path.join('data', 'logs', 'exp1', model + '_time' + '.csv'), 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow([model + RUN_TEST + '-' + t + '-' + str(depth) + '-' + str(units)] + time_callback.times)


def main():
    model = 'lstm'
    RUN_TEST = ''
    saved_model = None
    class_limit = None  # int, can be 1-101 or None
    seq_length = 20
    batch_size = 32
    nb_epoch = 1000
    delta = 5

    # Chose images or features and image shape based on network.
    if model in ['lstm', 'BiLSTM', 'ResLSTM', 'DenseLSTM', 'Dense_Bi_LSTM']:
        data_type = 'future_seq'
    else:
        raise ValueError("Invalid model. See train.py for options.")

    for _ in range(2):
        for seq_or_last in [True, False]:
            # for seq_or_last in [False]:
            for depth in range(1, 4):
                # for depth in range(1, 3):
                # for units in [2**i for i in range(11, 12)]:
                for units in [2**i for i in range(7, 12)]:
                    print('Training ', model, ': ', 'seq ' if seq_or_last else 'last ', depth, ' ', units)
                    train(data_type, seq_length, model, units=units, depth=depth, seq_or_last=seq_or_last, saved_model=saved_model, class_limit=class_limit, batch_size=batch_size, nb_epoch=nb_epoch, RUN_TEST=RUN_TEST, delta=delta)


if __name__ == '__main__':
    main()
