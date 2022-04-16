import os
import shutil
from pandas import DataFrame
from sklearn.metrics import classification_report
from statistics import mode
from collections import Counter
from alphabet_mode_main import predict_labels_from_frames

PATH_TO_VIDEOS = './Letters/Videos'
PATH_TO_FRAMES = './Letters/Frames'
PATH_TO_RESULTS = './Letters/results.csv'


def clean_dirs():
    for root, dirs, files in os.walk('./Letters/Frames'):
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
    if not pred or len( pred ) == 0:
        return ' '
    pred_final = Counter( pred ).most_common( 1 )[0][0]
    return pred_final



clean_dirs()

# Initialise the prediction array
arrPred = []

# Get list of test videos
list_of_videos = listdir_nohidden(PATH_TO_VIDEOS)

# Folder to save hand frames
if not os.path.exists(PATH_TO_FRAMES):
    os.makedirs( PATH_TO_FRAMES )

# Initialise predicted array
predicted = []

for video in list_of_videos:

    path_to_file = PATH_TO_VIDEOS + '/' + video
    video_name = video.split('.')[0]
    print('Test video ' + video + ' loaded')
    
    os.system('python ./hand_extractor/hand_extractor.py --source=%s --video=%s --frame_path=%s' % (path_to_file, video_name, PATH_TO_FRAMES) )
    
    arrPred = predict_labels_from_frames(PATH_TO_FRAMES + '/' + video_name)
    
    # Calculate Prediction and handle none cases
    try:
        valPred = mode(arrPred)
    except:
        valPred = ''
    
    actualLabel = video_name
    print("\nActual Value: " + actualLabel + " Predicted Value: " + valPred)
    predicted.append([valPred, actualLabel])

df = DataFrame (predicted, columns=['predicted', 'actual'])
print(classification_report(df.predicted, df.actual))
df.to_csv(PATH_TO_RESULTS)