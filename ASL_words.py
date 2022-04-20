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


# get List of unhidden files
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def final_prediction(pred):
    """ Returns the most common label from video frames as the final prediction """
    # print( pred )
    if not pred or len(pred) == 0:
        return ' '
    pred_final = Counter(pred).most_common(1)[0][0]
    return pred_final



clean_dirs()

# Folder to save hand frames
if not os.path.exists(PATH_TO_FRAMES):
    os.makedirs(PATH_TO_FRAMES)
if not os.path.exists(PATH_TO_HAND_FRAMES):
    os.makedirs(PATH_TO_HAND_FRAMES)

# Initialise the prediction array
arrPred = []

os.system('python ./posenet/Frames_Extractor.py --path_to_videos=%s --path_to_frames=%s' % (PATH_TO_VIDEOS, PATH_TO_FRAMES))
os.system('node ./posenet/scale_to_videos.js %s' % (PATH_TO_FRAMES))
os.system('python ./posenet/convert_to_csv.py --path_to_videos=%s --path_to_frames=%s' % (PATH_TO_VIDEOS, PATH_TO_FRAMES))

# Get list of test videos
list_of_videos = listdir_nohidden(PATH_TO_VIDEOS)

# Initialise predicted array
predicted = []

for video in list_of_videos:

    # if video != 'CAGE.mp4': continue

    path_to_file = PATH_TO_VIDEOS + '/' + video
    video_name = video.split('.')[0]
    print("Test video " + video + " loaded")

    os.system('python ./hand_extractor/hand_extractor.py --source=%s --video=%s --frame_path=%s' % (path_to_file, video_name, PATH_TO_HAND_FRAMES))
    # os.system('python ./hand_extractor.py --path_to_frames=%s --path_to_hand_frames=%s' % (PATH_TO_FRAMES + '/' + video_name, PATH_TO_HAND_FRAMES + '/' + video_name))

    keyptPosenet = pd.read_csv(PATH_TO_FRAMES + '/' + video_name + '/' + 'key_points.csv')

    coordRWx = keyptPosenet.rightWrist_x
    coordRWy = keyptPosenet.rightWrist_y
    coordLWx = keyptPosenet.leftWrist_x
    coordLWy = keyptPosenet.leftWrist_y

    letters = []
    lastframe = 0
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

    # frames = [[0, 182], [188, 288], [295, 525], [535, 640], [645, 870], [880, 980], [995, 1160]] # CAGE
    # frames = [[0, 170], [175, 266], [275, 489], [503, 587], [593, 756]] # EYE

    for i in range(len(frames)):
        # if i%2 == 0:
        prediction_frames = predict_words_from_frames_range(PATH_TO_HAND_FRAMES + '/' + video_name, frames[i][0], frames[i][1])
        prediction = final_prediction(prediction_frames)
        letters.append(prediction)

    predword = ''.join(letters).upper()
    actualLabel = video_name
    print("\nTrue Value: " + actualLabel + " Prediction: " + predword)
    predicted.append([predword, actualLabel])

df = DataFrame (predicted, columns=['predicted', 'actual'])
print(classification_report(df.predicted, df.actual))
df.to_csv(PATH_TO_RESULTS)
