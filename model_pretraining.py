import os
import shutil
from keras.utils.np_utils import normalize,to_categorical

from pandas import DataFrame
from sklearn.metrics import classification_report
from statistics import mode
from collections import Counter
from alphabet_mode_main import predict_labels_from_frames
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf
np.random.seed(123)


from handshape_feature_extractor import HandShapeFeatureExtractor
#
PATH_TO_VIDEOS = './Letters/Videos'
PATH_TO_FRAMES = './Letters/Frames'
PATH_TO_HAND_FRAMES = './Letters/Hand_Frames'
PATH_TO_RESULTS = './Letters/results.csv'
#
#
#
#
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

# os.system('python ./posenet/Frames_Extractor.py --path_to_videos=%s --path_to_frames=%s' % (PATH_TO_VIDEOS, PATH_TO_FRAMES))
# os.system('node ./posenet/scale_to_videos.js %s' % (PATH_TO_FRAMES))
# os.system('python ./posenet/convert_to_csv.py --path_to_videos=%s --path_to_frames=%s' % (PATH_TO_VIDEOS, PATH_TO_FRAMES))

# Get list of test videos
list_of_videos = listdir_nohidden(PATH_TO_VIDEOS)

# Initialise predicted array
predicted = []

training_data = []
retraining_data,labels = [], []
# For each of the videos generate frames and store
for video in list_of_videos:
    # We get the path to a file
    path_to_file = PATH_TO_VIDEOS + '/' + video
    video_name = video.split('.')[0]
    print('Test video ' + video + ' loaded')
    # Uses CNN to extract frames
    os.system('python ./hand_extractor/hand_extractor.py --source=%s --video=%s --frame_path=%s' % (path_to_file, video_name, PATH_TO_HAND_FRAMES))

    video_folder_path = PATH_TO_HAND_FRAMES + '/' + video_name
    path = os.path.join(video_folder_path, "*.png")
    # used to return all file paths that match a specific pattern
    frames = glob.glob(path)
    # print("Before printing frames",type(frames))
    # sort image frames
    frames.sort()
    model = HandShapeFeatureExtractor.get_instance()
    files = frames

    vectors = []
    video_names = []
    step = int(len(files) / 100)
    if step == 0:
        step = 1

    count = 0

    # going through all the frames one by one
    for video_frame in files:
        # print(video_frames)
        # assert len(video_frames) == 6
        img = cv2.imread(video_frame)
        # Convert image from one color space to another color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # results bunch of 66455e-06 8.5623396e-05 kind of data of particular shape
        results = model.process_feature(img)
        retraining_data.append(results)
        labels.append(ord(video_name)-65)
        video_names.append(os.path.basename(video_frame))
        count += 1
        if count % step == 0:
            # sys.stdout.write("-")
            sys.stdout.flush()

print("Restrainig")


print(retraining_data[0])
image_test = retraining_data[5]

print(len(retraining_data))
print(len(labels))

retraining_data = np.asarray(retraining_data)
print("Inside retraining data",retraining_data.shape)
labels = np.asarray(labels)
# print(labels[0].shape)
keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model
load_model = keras.models.load_model
BASE = os.path.dirname(os.path.abspath(__file__))
real_model = load_model(os.path.join(BASE, 'cnn_model.h5'))
for i, layer in enumerate (real_model.layers):
    print (i, layer)
    try:
        print ("    ",layer.activation)
    except AttributeError:
        print('   no activation attribute')

print(real_model.summary())

# # labels = np.asarray([1]*161)
print(retraining_data.shape)
print("Labels zero")
print(len(labels))
print(len(retraining_data))

# labels = to_categorical(labels, num_classes=27)
print(labels.shape)
print(labels)

randomize = np.arange(len(labels))

np.random.shuffle(randomize)
labelsMain = labels[randomize]
retraining_data_main = retraining_data[randomize]
print(len(labelsMain))
print(len(retraining_data_main))

history = real_model.fit(retraining_data_main, labelsMain, epochs=15, batch_size=16, validation_split=0.2)

print(history.history.keys())
print(history.history['loss'])
print(history.history['accuracy'])


# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# from keras.datasets import cifar10
#
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#
# # Preprocess the data (these are NumPy arrays)
# x_train = x_train.reshape(60000, 784).astype("float32") / 255
# x_test = x_test.reshape(10000, 784).astype("float32") / 255
#
# y_train = y_train.astype("float32")
# y_test = y_test.astype("float32")
#
# # Reserve 10,000 samples for validation
# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-10000]
# print("Final")
# print(type(y_train))
# print(len(y_train))
# print(y_train[0])
# import tensorflow as tf
# from tensorflow import keras
# inputs = keras.Input(shape=(784,), name="digits")
# x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
# x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
# outputs = keras.layers.Dense(10, activation="softmax", name="predictions")(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
#
# model.compile(
#     optimizer=keras.optimizers.RMSprop(),  # Optimizer
#     # Loss function to minimize
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     # List of metrics to monitor
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )
#
# print("Fit model on training data")
# print(type(y_train))
# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     epochs=2,
#     # We pass some validation for
#     # monitoring validation loss and metrics
#     # at the end of each epoch
#     validation_data=(x_val, y_val),
# )
# print(history.history.keys())
# print(history.history['sparse_categorical_accuracy'])
# print(history.history['val_sparse_categorical_accuracy'])
# print(history.history['loss'])
# print(history.history['val_loss'])



(xtrain,ytrain),(xTest,ytest) = cifar10.load_data();

ytrain = to_categorical(ytrain)
# print(ytrain)

