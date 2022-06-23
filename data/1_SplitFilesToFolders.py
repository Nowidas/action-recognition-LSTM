import os


def get_train_test_lists(ver='01'):
    train_file = os.path.join('ucfTrainTestlist', 'trainlist' + ver + '.txt')
    test_file = os.path.join('ucfTrainTestlist', 'testlist' + ver + '.txt')
    # create list from trainlistXX.txt
    with open(train_file) as f:
        train_list = [row.strip() for row in list(f)]
        train_list = [row.split(' ')[0] for row in train_list]
    # create list from testlistXX.txt
    with open(test_file) as f:
        test_list = [row.strip() for row in list(f)]
    return {'train': train_list, 'test': test_list}


def move_files(train_test_lists):
    # for group: test, train
    for group, videos in train_test_lists.items():
        # for evry video in group
        for video in videos:
            classname, videoname = video.split('/')
            videpath = os.path.join('UCF101', videoname)
            # create folder if not alredy created
            if not os.path.exists(os.path.join(group, classname)):
                os.mkdir(os.path.join(group, classname))
            # check if video exist
            if not os.path.exists(videpath):
                print(f'Video: {videoname} missing')
                continue
            # move file to destionation folder
            destionation = os.path.join(group, classname, videoname)
            print(f'Moving {videoname} to {destionation}')
            os.rename(videpath, destionation)


def main():
    move_files(get_train_test_lists())


if __name__ == '__main__':
    main()
