import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-dataset',
    '--path_to_dataset',
    dest='path_to_dataset',
    default=None,
    help='give a dataset folder path')
parser.add_argument(
    '-savemodel',
    '--save_model',
    dest='save_model',
    default=None,
    help='save model')
parser.add_argument(
    '-loadmodel',
    '--load_model',
    dest='load_model',
    default=None,
    help='load model')
args = parser.parse_args()

"""# Preprocessing the image data"""

#taking the train validation ratio as 4:1

batch_size=32
img_height=256
img_width=256


train_ds = tf.keras.utils.image_dataset_from_directory(
  args.path_to_dataset,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  args.path_to_dataset,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

"""# Modelling and training"""

#modelling

if args.load_model == None:
  model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26,activation='softmax')
  ])
  model.summary()
else:
  print('\nLoading Model: ' + args.load_model)
  model = tf.keras.models.load_model(args.load_model)

if args.load_model == None:
  print('\n**********Training on Kaggle datatset**********\n')
else:
  print('\n**********Training on our datatset**********\n')

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_ds, batch_size=32,validation_batch_size=32, validation_data=test_ds,epochs=3)

model.save(args.save_model)

