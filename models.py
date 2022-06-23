from operator import concat
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Bidirectional, InputLayer, DropoutWrapper, ResidualWrapper, LSTMCell, RNN, add, MaxPool1D, Concatenate, AveragePooling1D, Reshape
from keras.layers.recurrent import LSTM
from tensorflow.keras.optimizers import Adam
import sys
import tensorflow as tf
import keras
import time

from data import DataSet


class Model():
    def __init__(self, nb_classes, model, seq_length=30, saved_model=None, features_length=2048, units=None, depth=None, ret_seq=False, dropout=None, recurent_dropout=None):
        """
        `model` = one of:
            lstm
            BiLSTM
            ResLSTM
            DenseLSTM
            Dense_Bi_LSTM
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences (same as for DataSet)
        `saved_model` = the path to a saved Keras model to load
        `features_length` = lenght of extracted feature fector for evry image 
        `units` = nb. of LSTM units in model (if used)
        `depth` = nb. of LSTM layer depth in model (if used)
        `ret_seq` = technique for model output:  deafult return last output of lstm node (if used)
        `dropout` = set dropout in LSTM layer
        `recurent_dropout` = set recurent_dropout in LSTM layer
        
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm' or model == 'LSTM':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm(units, depth, ret_seq, dropout, recurent_dropout)
        elif model == 'BiLSTM':
            print("Loading BiLSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.BiLSTM(units)
        elif model == 'ResLSTM':
            print("Loading ResLSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.ResLSTM()
        elif model == 'DenseLSTM':
            print("Loading DenseLSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.DenseLSTM()
        elif model == 'Dense_Bi_LSTM':
            print("Loading Dense_Bi_LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.Dense_Bi_LSTM()
        else:
            print("Unknown network.")
            sys.exit()

        # Compile the network.
        optimizer = Adam(learning_rate=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

        print(self.model.summary())

    def lstm(self, units, depth, ret_seq, dropout, recurent_dropout):
        """Plain LSTM network"""
        units = 2048 if units == None else units
        depth = 1 if depth == None else depth
        ret_seq = True if ret_seq == True else ret_seq
        dropout = 0.5 if dropout == None else dropout
        recurent_dropout = 0 if recurent_dropout == None else recurent_dropout

        # Model.
        model = Sequential()
        model.add(InputLayer(input_shape=self.input_shape))
        for _ in range(depth - 1):
            model.add(RNN(LSTMCell(units, dropout=dropout), return_sequences=True))
        model.add(RNN(LSTMCell(units, dropout=dropout), return_sequences=ret_seq))
        if ret_seq:
            model.add(AveragePooling1D(pool_size=self.input_shape[0]))
            model.add(Reshape((units, )))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def BiLSTM(self, units):
        units = 1024 if units == None else units

        # Model.
        inputs = keras.Input(shape=self.input_shape, name='futures')
        lstm1 = RNN(LSTMCell(units, dropout=0.5), return_sequences=False)(inputs)
        lstm2 = RNN(LSTMCell(units, dropout=0.5), return_sequences=False, go_backwards=True)(inputs)
        block_1_output = Concatenate()([lstm1, lstm2])
        x = Dense(1024, activation='relu')(block_1_output)
        x = Dropout(0.5)(x)
        outputs = Dense(self.nb_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        return model

    def ResLSTM(self):
        # Model.
        inputs = keras.Input(shape=self.input_shape, name='futures')
        x = RNN(LSTMCell(2048, dropout=0.5), return_sequences=True)(inputs)
        block_1_output = add([x, inputs])
        x = RNN(LSTMCell(2048, dropout=0.5), return_sequences=True)(block_1_output)
        block_2_output = add([x, block_1_output])
        x = AveragePooling1D(pool_size=self.input_shape[0])(block_2_output)
        x = Reshape((2048, ))(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.nb_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        return model

    def DenseLSTM(self):
        # Model.
        inputs = keras.Input(shape=self.input_shape, name='futures')
        lstm1 = RNN(LSTMCell(2048, dropout=0.5), return_sequences=True)(inputs)
        block_1_output = Concatenate()([lstm1, inputs])
        lstm2 = RNN(LSTMCell(2 * 2048, dropout=0.5), return_sequences=True)(block_1_output)
        block_2_output = Concatenate()([lstm1, lstm2, block_1_output])
        x = AveragePooling1D(pool_size=self.input_shape[0])(block_2_output)
        x = Reshape((5 * 2048, ))(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.nb_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        return model

    def Dense_Bi_LSTM(self):
        # use: units = 1024
        units = 1024 if units == None else units

        # Model.
        inputs = keras.Input(shape=self.input_shape, name='futures')
        avg = AveragePooling1D(pool_size=self.input_shape[1] // units, data_format='channels_first')(inputs)

        lstm1 = RNN(LSTMCell(units, dropout=0.5), return_sequences=True)(avg)
        block_1_output = Concatenate()([lstm1, avg])
        lstm2 = RNN(LSTMCell(2 * units, dropout=0.5), return_sequences=True)(block_1_output)
        block_2_output = Concatenate()([lstm1, lstm2, block_1_output])
        x = AveragePooling1D(pool_size=self.input_shape[0])(block_2_output)
        forward_layer = Reshape((5 * units, ))(x)

        lstm1 = RNN(LSTMCell(units, dropout=0.5), return_sequences=True, go_backwards=True)(avg)
        block_1_output = Concatenate()([lstm1, avg])
        lstm2 = RNN(LSTMCell(2 * units, dropout=0.5), return_sequences=True, go_backwards=True)(block_1_output)
        block_2_output = Concatenate()([lstm1, lstm2, block_1_output])
        x = AveragePooling1D(pool_size=self.input_shape[0])(block_2_output)
        backward_layer = Reshape((5 * units, ))(x)

        x = Concatenate()([forward_layer, backward_layer])

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.nb_classes, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        return model


def main():
    model = Model(101, 'Dense_Bi_LSTM')
    # Plot model
    tf.keras.utils.plot_model(model.model, '7.png', show_shapes=True, dpi=300, show_layer_names=False)
    DataSet()


if __name__ == '__main__':
    main()
