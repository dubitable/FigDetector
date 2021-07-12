#Main library
import tensorflow as tf

#Helper libraries
import os, pickle, random
import numpy as np
import pandas as pd

numpy_dir = "numpy"

class_names = sorted([file.split(".")[0] for file in os.listdir(numpy_dir)])
figs = [class_name for class_name in class_names if "fig" in class_name]
print(class_names)
print(figs)

images = []
labels = []

for file in os.listdir(numpy_dir):
  file_path = "{}/{}".format(numpy_dir, file)
  array = (np.load(file_path, allow_pickle = True) / 255)
  print(file)
  for image in array:
    images.append(image)
    labels.append(class_names.index(file.split(".")[0]))
    
temp = list(zip(images, labels))
np.random.shuffle(temp)
images, labels = zip(*temp)
images, labels = np.array(images, dtype=object), np.array(labels)

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(50, 50)),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dense(len(class_names))
])


model.compile(optimizer="adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
                                                 
model.fit(images, labels, epochs=10, callbacks=[cp_callback]))