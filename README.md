# CSE535-ASLFingerSpelling

## How to run

You need to have `node` and `python` installed on your environment to run this code.<br>

1. Create the following directories `Letters/Videos`, `Words/Videos` and `asl-alphabet/asl_alphabet_train`.
2. Put the videos of your alphabets in the directory `Letters/Videos/{your_name}/` and keep the video name as the alphabet name.<br>
   Eg: `Letters/Videos/Bob/A.mp4`, `Letters/Videos/Alice/A.mp4`
3. Put the videos of the words in the directory `Words/Videos/` and keep the video name as the word name.<br>
   Eg: `Words/Videos/ACE.mp4`
4. Put the training dataset from Kaggle (https://www.kaggle.com/grassknoted/asl-alphabet) in the directory `asl-alphabet/asl_alphabet_train/`.
5. Install the requirements using the command `pip install -r requirements.txt`.
6. Install the `node_modules` in the `posenet/` directory using the command `npm install`.
7. Run the file `ASL_letters.py` which will extract hand frames from the alphabet videos using posenet, train the CNN model and predict the alphabets from the hand frames.
8. Run the file `ASL_words.py` which will extract hand frames from the word videos using posenet and predict the words from the hand frames.