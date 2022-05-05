import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-videos',
    '--path_to_videos',
    dest='path_to_videos',
    default=0,
    help='give a videos folder path')
parser.add_argument(
    '-frames',
    '--path_to_frames',
    dest='path_to_frames',
    default=0,
    help='give a frames folder path')
args = parser.parse_args()

# Path to the directory containing Video Files
path_to_videos = args.path_to_videos
path_to_frames = args.path_to_frames
video_files = os.listdir(path_to_videos)

for file in video_files:
    video = cv2.VideoCapture(path_to_videos + '/' + file)
    # flip =True
    count = 0
    success = 1
    arr_img = []
    # If such a directory doesn't exist, creates one and stores its Images
    if not os.path.isdir(path_to_frames + '/' + os.path.splitext(file)[0] + '/'):
        os.mkdir(path_to_frames + '/' + os.path.splitext(file)[0])
        new_path = path_to_frames + '/' + os.path.splitext(file)[0]
        while success:
            success, image = video.read()
            # Frames when generated are getting rotated clockwise by above method, so correcting it
            # if flip:
                # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            arr_img.append(image)
            count += 1
        # Sub sampling the number of frames
        # numbers = sorted(random.sample(range(len(arr_img)), 45))
        count = 0
        for i in range(len(arr_img)):
            if arr_img[i] is not None:
                cv2.imwrite(new_path + "/%d.png" % count, arr_img[i])
                count += 1
    print('Frames extracted for file: {0}'.format(file))
