'''
predict func for given saved model
'''
from keras.models import Model, load_model
from data import DataSet
import numpy as np
import cv2
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from extracting_futures_ff import build_feature_model

feature_model = build_feature_model()


def predict(saved_model, video_name):

    # load data from file name
    DELTA = int(saved_model[saved_model.find("_d") + 2:saved_model.find("_s")])
    SEQ_LENGHT = int(saved_model[saved_model.find("_s") + 2:saved_model.find("_e")])

    data = DataSet(seq_length_in_frames=SEQ_LENGHT, delta=DELTA)
    prediction_model = load_model(saved_model)
    vidcap = cv2.VideoCapture(video_name)

    count = 0
    frame_set = []
    frame_prediction = []

    success, image = vidcap.read()
    while success:
        if count % DELTA == 0:
            cv2.imwrite('read_img.jpg', image)  # save frame as JPEG file
            features = extract_features_from_image()
            frame_set.append(features)
        if len(frame_set) == SEQ_LENGHT:
            prediction = prediction_model.predict(np.expand_dims(frame_set, axis=0))
            most_prediceted = np.argmax(prediction, axis=1)
            string_prediction = data.classes[most_prediceted[0]] + ' ' + str(round(prediction[0][most_prediceted[0]] * 100, 2)) + '%'
            print(string_prediction)
            frame_prediction.append((count, string_prediction))
            frame_set = []
        success, image = vidcap.read()
        count += 1
    return frame_prediction


def extract_features_from_image():

    h, w = (299, 299)
    img = image.load_img('read_img.jpg', target_size=(h, w))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = feature_model.predict(x)
    features = np.squeeze(features, axis=0)
    return features


def main():
    print(predict(1, 1))


if __name__ == '__main__':
    main()
