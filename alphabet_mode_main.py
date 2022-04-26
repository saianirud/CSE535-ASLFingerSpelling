import glob
import numpy as np
import os
import tensorflow as tf
from handshape_feature_extractor import HandShapeFeatureExtractor


def get_inference_vector_one_frame_alphabet(files_list):
    # model trained based on https://www.kaggle.com/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out

    model = HandShapeFeatureExtractor()
    vectors = []

    for video_frame in files_list:
        results = model.extract_feature(video_frame)
        vectors.append(np.argmax(results))

    return vectors

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def load_label_dicts(label_file):
    id_to_labels = load_labels(label_file)
    labels_to_id = {}
    i = 0

    for id in id_to_labels:
        labels_to_id[id] = i
        i += 1

    return id_to_labels, labels_to_id


def predict_labels_from_frames(video_folder_path):
    
    files = []
    # wildcard to select all frames for given video file
    
    path = os.path.join(video_folder_path, "*.png")
    frames = glob.glob(path)
    
    # sort image frames
    frames.sort()
    files = frames
    
    prediction_vector = get_inference_vector_one_frame_alphabet(files)
    
    label_file = 'output_labels_alphabet.txt'
    id_to_labels, labels_to_id = load_label_dicts(label_file)
    
    final_predictions=[]
    
    for i in range(len(prediction_vector)):
        for ins in labels_to_id:
            if prediction_vector[i] == labels_to_id[ins]:
                final_predictions.append(ins)
    
    return final_predictions

def predict_words_from_frames(video_folder_path, start, end):
    
    files = []
    # wildcard to select all frames for given video file
    
    path = os.path.join(video_folder_path, "*.png")
    frames = glob.glob(path)
    names_arr = [video_folder_path + '\\' + str(i) + '.png' for i in range(start, end+1)]

    files = [frame for frame in frames if frame in names_arr]
    
    # sort image frames
    files.sort()
    # files = frames[start:end]
    
    prediction_vector = get_inference_vector_one_frame_alphabet(files)
    
    label_file = 'output_labels_alphabet.txt'
    id_to_labels, labels_to_id = load_label_dicts(label_file)
    
    final_predictions=[]
    
    for i in range(len(prediction_vector)):
        for ins in labels_to_id:
            if prediction_vector[i] == labels_to_id[ins]:
                final_predictions.append(ins)
    
    return final_predictions