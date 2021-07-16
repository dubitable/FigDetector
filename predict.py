#Main library
import tensorflow as tf

#Helper libraries
import os
import numpy as np

checkpoint_dir = "model1/"
class_names = ['apple', 'apricot', 'avocado', 'banana', 'bell pepper', 'black berry', 'blueberry', 'cantaloupe', 'cherry', 
               'coconut', 'coffee', 'desert fig', 'eggplant', 'fig', 'grape', 'grapefruit', 'kiwi', 'lemon', 'lime', 'lychee', 
               'mango', 'orange', 'pear', 'pineapple', 'pomegranate', 'pumpkin', 'raspberry', 'strawberry', 'watermelon']

def load_model(dir):
  model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(29)
  ])

  model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  latest = tf.train.latest_checkpoint(dir)
  model.load_weights(latest)
  return model

def predict(path, model):
  img = tf.keras.preprocessing.image.load_img(path, target_size=(180, 180))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  return score

img_path = input("path = ")
model = load_model(checkpoint_dir)

prediction = predict(img_path, model)
print("Got {} with {}% confidence.".format(class_names[np.argmax(prediction)], max(prediction) * 100 ))