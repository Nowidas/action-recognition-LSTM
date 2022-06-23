"""
Given a saved model file, plot and input to file confusion matrix for every class.
"""
from keras.models import load_model
import pandas as pd
from data import DataSet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d.
    # Must be a weights file.
    saved_model = 'data/checkpoints/lstm__matrix_d5_s5_e012_l0.948.hdf5'
    # Sequence length must match the lengh used during training.
    seq_length = 5
    delta = None
    # Limit must match that used during training.
    class_limit = None

    model = load_model(saved_model)

    # Get the data and process it.
    data = DataSet(seq_length_in_frames=seq_length, class_limit=class_limit, delta=delta)
    train, test = data.get_train_test_split()
    # Extract the sample from the data.
    # sample = data.get_frames_by_filename(video_name, data_type)
    y_pred, y_true = [], []
    # Predict!
    for vid in test:
        seq = data.get_all_extracted_sequence(vid)
        for s in seq:
            prediction = model.predict(np.expand_dims(s, axis=0))
            most_prediceted = np.argmax(prediction, axis=1)
            y_pred.append(most_prediceted[0])
            y_true.append(data.classes.index(vid[1]))

    test_acc = sum(el_pred == el_true for el_pred, el_true in zip(y_pred, y_true)) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    print(confusion_mtx)

    t_np = confusion_mtx.numpy()  # convert to Numpy array
    df = pd.DataFrame(t_np)  # convert to a dataframe
    df.to_csv("matrix1.csv", index=False)  # save to file

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=data.classes, yticklabels=data.classes, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


if __name__ == '__main__':
    main()
