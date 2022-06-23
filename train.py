"""
Train LSTMs on extracted features
"""

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import Model
from data import DataSet
import time
import os.path
import tensorflow as tf


def train(data_type, seq_length, model, saved_model=None, class_limit=None, batch_size=32, nb_epoch=100, initial_epoch=0, RUN_TEST='', delta=5):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(filepath=os.path.join('data', 'checkpoints', model + '_' + RUN_TEST + '_d' + str(delta) + '_s' + str(seq_length) + '_e{epoch:03d}_l{val_loss:.3f}.hdf5'), verbose=1, save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model + RUN_TEST))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + RUN_TEST + '-' + 'training-' + str(timestamp) + '.log'))

    # Get the data and process it.
    data = DataSet(seq_length_in_frames=seq_length, class_limit=class_limit, delta=delta)

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    # steps_per_epoch = (len(data.data) * 0.7) // batch_size
    # steps_per_epoch = (96000 * 0.7) // batch_size  # 5_5
    # steps_per_epoch = (24000 * 0.7) // batch_size  # 5_20
    steps_per_epoch = 43917 // batch_size  # 2_25
    # data_len = data.calculate_data_len() #or just calculate data_len (longer takes)
    # print('LENGHT:', data_length)

    # Get generators.
    generator = data.generator_sequence_and_return_X_Y(data_type, 'train', batch_size)
    val_generator = data.generator_sequence_and_return_X_Y(data_type, 'test', batch_size)

    # Get the model.
    rm = Model(len(data.classes), model, seq_length, saved_model)

    # Plot model
    # tf.keras.utils.plot_model(
    #     os.path.join('graph', rm.model, model + RUN_TEST + '.png'), show_shapes=True, dpi=300)

    # Use fit generator.
    rm.model.fit(x=generator, steps_per_epoch=steps_per_epoch, epochs=nb_epoch, verbose=1, callbacks=[tb, early_stopper, csv_logger, checkpointer], validation_data=val_generator, validation_steps=30, workers=4, initial_epoch=initial_epoch)


def main():
    model = 'lstm'
    RUN_TEST = '30'
    saved_model = None
    initial_epoch_if_saved_model = 0
    class_limit = None  # int, can be 1-101 or None
    seq_length = 25
    batch_size = 32
    nb_epoch = 1000
    delta = 2

    if model in ['lstm', 'BiLSTM', 'ResLSTM', 'DenseLSTM', 'Dense_Bi_LSTM']:
        data_type = 'future_seq'
    else:
        raise ValueError("Invalid model. See train.py for options.")

    for _ in range(5):
        train(data_type, seq_length, model, saved_model=saved_model, class_limit=class_limit, batch_size=batch_size, delta=delta, nb_epoch=nb_epoch, initial_epoch=initial_epoch_if_saved_model, RUN_TEST=RUN_TEST)
        RUN_TEST = str(int(RUN_TEST) + 1)


if __name__ == '__main__':
    main()
