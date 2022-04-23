"""# Prediction on the test data"""

import os
import numpy as np
import tensorflow as tf

batch_size=32
img_height=256
img_width=256

train_ds = tf.keras.utils.image_dataset_from_directory(
  "./asl-alphabet/asl_alphabet_train/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
 
# Get the list of all files and directories
path = "./asl-alphabet/asl_alphabet_test/"
dir_list = os.listdir(path)

model = tf.keras.models.load_model('cnn_model.h5')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

actual=[]
pred=[]
for i in dir_list:
    actual.append('E')
    test_image = tf.keras.utils.load_img('./asl-alphabet/asl_alphabet_test/'+i, target_size = (256, 256))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    pred.append(class_names[np.argmax(result)])

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


print("Test accuracy=",accuracy_score(pred,actual))
print("Classification report:\n",classification_report(pred,actual))