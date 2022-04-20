import cv2
import pandas as pd
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_frames',
        dest='path_to_frames',
        default=0,
        help='give a frames folder path')
    parser.add_argument(
        '--path_to_hand_frames',
        dest='path_to_hand_frames',
        default=0,
        help='give a hand frames folder path')
    args = parser.parse_args()

    path_to_frames = args.path_to_frames
    path_to_hand_frames = args.path_to_hand_frames

    pos_key = pd.read_csv(path_to_frames + '/' + 'key_points.csv')
    rightWrist_x = pos_key.rightWrist_x
    rightWrist_y = pos_key.rightWrist_y
    leftWrist_x = pos_key.leftWrist_x
    leftWrist_y = pos_key.leftWrist_y

    frames =  [file for file in os.listdir(path_to_frames) if file.endswith('.png')]
    files = sorted(frames, key=lambda x: int(os.path.splitext(x)[0]))
    i = 0

    if not os.path.isdir(path_to_hand_frames):
        os.mkdir(path_to_hand_frames)

    for video_frame in files:
        try:
            if i < len(rightWrist_x):
                image_path = path_to_frames + '/' + video_frame
                img = cv2.imread(image_path)
                cropped_image = img[round(rightWrist_y[i])-200:round(rightWrist_y[i])+50, round(rightWrist_x[i])-200:round(rightWrist_x[i])+125]
                # flipped_cropped_image = cv2.flip(cropped_image,1)
                image_path = path_to_hand_frames + '/' + str(i) + '.png'
                cv2.imwrite(image_path, cropped_image)
                i = i + 1
        except:
            i = i + 1
