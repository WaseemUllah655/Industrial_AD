# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:15:00 2023

@author: Waseem_Ullah
"""

import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
from keras.models import Model
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# Define the custom loss function
def anomaly_detection_loss(y_true, y_pred, threshold):
    # Compute the mean squared error
    mse = K.mean(K.square(y_true - y_pred), axis=[1,2,3])
    # Compute the standard deviation of the mean squared error
    mse_std = K.std(mse)
    # Compute the threshold value
    #threshold = mse_std + K.mean(mse)
    # Compute the binary cross-entropy loss
    bce_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[1,2,3])
    # Compute the anomaly loss
    anomaly_loss = K.switch(mse > threshold, bce_loss, 0.0)
    return anomaly_loss

def load_custom_dataset(directory):
    X = []
    y = []
    label_encoder = LabelEncoder()
    for subdir, _, files in os.walk(directory):
        for file in files:
            # Load the image and resize it to the desired shape
            image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))
            # Append the image and label to the X and y lists
            X.append(image)
            label = subdir.split("/")[-1]
            y.append(label)
    # Convert the labels to integer labels using LabelEncoder
    y = label_encoder.fit_transform(y)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y

train_dir = "./Dataset/Train/"
val_dir = "./Dataset/Validation/"
test_dir = "./Dataset/Test/"

# Load the custom dataset
x_train, y_train = load_custom_dataset(train_dir)
x_val, y_val = load_custom_dataset(val_dir)
x_test, y_test = load_custom_dataset(test_dir)

x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

input_shape = (28, 28, 1)

inputs = Input(shape=input_shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

model = Model(inputs=inputs, outputs=decoded)

threshold = 0.4
model.compile(optimizer='adam', loss=lambda y_true, y_pred: anomaly_detection_loss(y_true, y_pred, threshold))
model.summary()

history = model.fit(x_train, x_train,epochs=200, batch_size=128, validation_data=(x_val, x_val))
test_loss = model.evaluate(x_test, x_test, batch_size=128)
print("Test loss:", test_loss)
preds = model.predict(x_test)

mse = np.mean(np.square(x_test - preds), axis=(1, 2, 3))
bce = np.mean(-(x_test * np.log(preds + K.epsilon()) + (1 - x_test) * np.log(1 - preds + K.epsilon())), axis=(1, 2, 3))
anomaly_score = mse + bce
sns.histplot(anomaly_score, kde=True)
plt.xlabel("Anomaly score")
plt.show()



# Define the threshold value
threshold = 0.4

# Compute the anomaly score for the test set
preds = model.predict(x_test)
mse = np.mean(np.square(x_test - preds), axis=(1, 2, 3))
bce = np.mean(-(x_test * np.log(preds + K.epsilon()) + (1 - x_test) * np.log(1 - preds + K.epsilon())), axis=(1, 2, 3))
anomaly_score = mse + bce

# Create the binary mask
mask = (anomaly_score > threshold).astype(np.float32)

# Reshape the mask to match the last dimension of x_test
mask = mask.reshape((-1, 1, 1, 1))

# Apply the mask to the original images
anomaly_images = x_test * mask

# Plot the anomaly images as a heatmap
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i < len(anomaly_images):
        ax.imshow(anomaly_images[i, ..., 0], cmap='jet')
        ax.set_title(f"Anomaly score: {anomaly_score[i]:.2f}")
    ax.axis('off')
plt.tight_layout()
plt.show()