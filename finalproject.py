import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report

curr_dir = './'  # may need to change to atual current directory
data_dir = curr_dir + '/image_dataset/'

batch_size = 500
img_height = 200
img_width = 200

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=.2)

validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                        validation_split=.2)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    data_dir,  # This is the source directory for training images
    target_size=(200, 200),  # All images will be resized to 200x200
    batch_size=batch_size,
    subset="training",
    # Use binary labels
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    data_dir,  # This is the source directory for training images
    target_size=(200, 200),  # All images will be resized to 200x200
    batch_size=batch_size,
    subset="validation",
    # Use binary labels
    class_mode='binary',
    shuffle=False)

num_classes = 2

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2)
])

model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 15
history = model.fit(
    train_generator,
    callbacks=[callback],
    validation_data=validation_generator,
    epochs=epochs,
    verbose=1,
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

validation_generator.reset()
preds = model.predict(validation_generator, verbose=1)
y_pred = tf.argmax(preds, axis=1)

sns.heatmap(confusion_matrix(validation_generator.classes, y_pred), annot=True)
plt.show()

print(classification_report(validation_generator.classes, y_pred))

fpr, tpr, _ = roc_curve(validation_generator.classes, y_pred)

roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()