import os
import math
import shutil
import pandas as pd
from pandas import DataFrame
from statistics import mode
from alphabet_mode_main import predict_words_from_frames
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

PATH_TO_VIDEOS = './Words/Videos'
PATH_TO_FRAMES = './Words/Frames'
PATH_TO_HAND_FRAMES = './Words/Hand_Frames'
PATH_TO_RESULTS = './Words/results.csv'


def clean_dirs():
    for root, dirs, files in os.walk(PATH_TO_FRAMES):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    for root, dirs, files in os.walk(PATH_TO_HAND_FRAMES):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def generate_posenet_keypoints():
    print('\n**********Generating Posenet Keypoints for Videos**********\n')
    os.system('python ./posenet/Frames_Extractor.py --path_to_videos=%s --path_to_frames=%s' % (PATH_TO_VIDEOS, PATH_TO_FRAMES))
    os.system('node ./posenet/scale_to_videos.js %s' % (PATH_TO_FRAMES))
    os.system('python ./posenet/convert_to_csv.py --path_to_videos=%s --path_to_frames=%s' % (PATH_TO_VIDEOS, PATH_TO_FRAMES))


def segment_videos(video_name):

    print('\n********** Segmenting Video: {0} **********\n'.format(video_name))
    keyptPosenet = pd.read_csv(PATH_TO_FRAMES + '/' + video_name + '/' + 'key_points.csv')

    coordRWx = keyptPosenet.rightWrist_x
    coordRWy = keyptPosenet.rightWrist_y
    threshold = 20
    frame_arr = []

    for i in range(keyptPosenet.shape[0]-1):
        dist = math.sqrt( ((coordRWx[i + 1] - coordRWx[i]) ** 2) + ((coordRWy[i + 1] - coordRWy[i]) ** 2))
        if dist < threshold and coordRWy[i] < 600:
            frame_arr.append(i)
    
    frames = []
    start = 0
    end = start

    for i in range(1, len(frame_arr)):
        diff = frame_arr[i] - frame_arr[i-1]
        if diff == 1 or diff == 2 or diff == 3:
            end = frame_arr[i]
            continue

        if (end - start) >= 50:
            frames.append([start, end])
        start = frame_arr[i]
        end = start
    
    if (end - start) >= 50:
            frames.append([start, end])

    print('Frames: ', frames)

    return frames


def predict_word(frames, video_name): 
    print('\n********** Predict Video: {0} **********\n'.format(video_name))
    letters = []
    for i in range(len(frames)):
        prediction_frames = predict_words_from_frames(PATH_TO_HAND_FRAMES + '/' + video_name, frames[i][0], frames[i][1])
        prediction = mode(prediction_frames)
        letters.append(prediction)

    return ''.join(letters).upper()


def classification_report(df):
    accuracy = []
    for index, row in df.iterrows():
        count = 0
        for i in range(min(len(row['ground_truth']), len(row['predicted']))):
            if row['ground_truth'][i] == row['predicted'][i]:
                count += 1
        accuracy.append(count * 100 / len(row['ground_truth']))
    df['accuracy (%)'] = accuracy
    return df



clean_dirs()

# Folder to save hand frames
if not os.path.exists(PATH_TO_FRAMES):
    os.makedirs(PATH_TO_FRAMES)
if not os.path.exists(PATH_TO_HAND_FRAMES):
    os.makedirs(PATH_TO_HAND_FRAMES)

# Initialise predicted array
output = []

# Create posenet wrist points
generate_posenet_keypoints()

for root, dirs, files in os.walk(PATH_TO_VIDEOS):
    for video in files:

        print('\n' + '-'*100)

        path_to_file = PATH_TO_VIDEOS + '/' + video
        video_name = video.split('.')[0]

        print('\n********** Extracting Hand Frames for Video: {0} **********\n'.format(video))
        # os.system('python ./hand_extractor.py --path_to_frames=%s --path_to_hand_frames=%s' % (PATH_TO_FRAMES + '/' + video_name, PATH_TO_HAND_FRAMES + '/' + video_name))
        print('Hand Frames extracted for Video: {0}'.format(video))

        frames = segment_videos(video_name)

        prediction = predict_word(frames, video_name)
        print('Prediction: {0}\tGround Truth: {1}'.format(prediction, video_name))
        output.append([prediction, video_name])


df = DataFrame(output, columns=['predicted', 'ground_truth'])
print('\n' + '-'*100)
df = classification_report(df)
print(df)
print('\nAverage Accuracy: {0}\n'.format(df['accuracy (%)'].mean()))
df.to_csv(PATH_TO_RESULTS)
