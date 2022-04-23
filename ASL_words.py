import os
import math
import shutil
import pandas as pd
from pandas import DataFrame
import time
from sklearn.metrics import classification_report
from statistics import mode
from collections import Counter
from alphabet_mode_main import predict_labels_from_frames, predict_words_from_frames, predict_words_from_frames_range

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

    print('\n**********Segmenting Video: {0}**********\n'.format(video_name))
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


def final_prediction(pred):
    """ Returns the most common label from video frames as the final prediction """
    # print( pred )
    if not pred or len(pred) == 0:
        return ' '
    pred_final = Counter(pred).most_common(1)[0][0]
    return pred_final


def predict_word(frames, video_name): 
    print('\n**********Predict Video: {0}**********\n'.format(video_name))
    letters = []
    for i in range(len(frames)):
        prediction_frames = predict_words_from_frames_range(PATH_TO_HAND_FRAMES + '/' + video_name, frames[i][0], frames[i][1])
        prediction = final_prediction(prediction_frames)
        letters.append(prediction)

    return ''.join(letters).upper()



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

        # if video != 'ZIP.mp4': continue

        path_to_file = PATH_TO_VIDEOS + '/' + video
        video_name = video.split('.')[0]

        # print('\n**********Extracting Hand Frames for Video: {0}**********\n'.format(video))
        # os.system('python ./hand_extractor/hand_extractor.py --source=%s --video=%s --frame_path=%s' % (path_to_file, video_name, PATH_TO_HAND_FRAMES))
        os.system('python ./hand_extractor.py --path_to_frames=%s --path_to_hand_frames=%s' % (PATH_TO_FRAMES + '/' + video_name, PATH_TO_HAND_FRAMES + '/' + video_name))

        frames = segment_videos(video_name)

        # frames = [[0, 160], [250, 480], [570, 726]] # BAD
        # frames = [[0, 180], [260, 450], [550, 757]] # HAT
        # frames = [[0, 130], [220, 410], [490, 686]] # HAT

        prediction = predict_word(frames, video_name)
        print('\nPrediction: {0}\tGround Truth: {1}\n'.format(prediction, video_name))
        output.append([prediction, video_name])


df = DataFrame(output, columns=['predicted', 'actual'])
print(classification_report(df.predicted, df.actual))
df.to_csv(PATH_TO_RESULTS)
