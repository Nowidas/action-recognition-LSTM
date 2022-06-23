import os
import glob
import cv2
import csv


def split_to_frames():
    data_csv = []
    for folder in ['test', 'train']:
        class_folders = glob.glob(os.path.join(folder, '*'))
        for video_class in class_folders:
            videos_in_class = glob.glob(os.path.join(video_class, '*.avi'))
            for video_path in videos_in_class:
                video_info = get_video_info(video_path)
                train_or_test, classname, videoname_extention, videoname = video_info
                if not os.path.exists(os.path.join(train_or_test, classname, videoname + '_f0001.jpg')):
                    source = os.path.join(
                        train_or_test, classname, videoname_extention)
                    vidcap = cv2.VideoCapture(source)
                    success, image = vidcap.read()
                    count = 0
                    while success:
                        # save frame as JPEG file
                        cv2.imwrite(
                            os.path.join(train_or_test, classname, videoname + '_f%04d.jpg' % count), image)
                        success, image = vidcap.read()
                        count += 1
                    data_csv.append(
                        [train_or_test, classname, videoname, count])
                    print("Generated %d frames for %s" % (count, videoname))
    with open('data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_csv)
    print("Splited %d video files." % (len(data_csv)))


def get_video_info(video_path):
    """returns train_or_test, classname, videoname_extention, videoname"""
    dirs = video_path.split(os.path.sep)
    train_or_test = dirs[0]
    classname = dirs[1]
    videoname_extention = dirs[2]
    videoname = videoname_extention.split('.')[0]

    return train_or_test, classname, videoname_extention, videoname


def main():
    split_to_frames()


if __name__ == '__main__':
    main()
