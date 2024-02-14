# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:49:10 2023

@author: Waseem_Ullah
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths to the normal and anomalous image directories
normal_path = './input/'
anomalous_path = './input/'

# Define the image size and number of channels
img_height, img_width = 256, 256
channels = 3
# Load the normal images and resize them
normal_datagen = ImageDataGenerator(rescale=1./255)
normal_generator = normal_datagen.flow_from_directory(
    normal_path,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary',
    classes=['Normal'],
    color_mode='rgb'
    )

# Load the anomalous images and resize them
anomalous_datagen = ImageDataGenerator(rescale=1./255)
anomalous_generator = anomalous_datagen.flow_from_directory(
    anomalous_path,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary',
    classes=['Abnormal'],
    color_mode='rgb'
    )

# print(len(anomalous_generator))
# print(len(normal_generator))

# Define the input shape
input_shape = (img_height, img_width, channels)

# Define the encoder architecture
encoder_input = layers.Input(shape=input_shape)
encoder = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)
encoder = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder)
encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)
encoder = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoder)
encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)
encoder_output = layers.Flatten()(encoder)

# Define the one-class SVM layer
svm_layer = layers.Dense(1, activation='sigmoid')(encoder_output)

# Define the anomaly detection model
model = keras.models.Model(inputs=encoder_input, outputs=svm_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
# Train the model on the normal images
model.fit(normal_generator, epochs=5, shuffle=True)

# Evaluate the model on the anomalous images
anomaly_scores = model.predict(anomalous_generator)

# Generate the heatmap for the anomalous images
anomaly_heatmap = tf.reduce_mean(tf.square(anomaly_scores - 0.5), axis=-1)


import matplotlib.pyplot as plt
import numpy as np

# Reshape the anomaly heatmap to match the image shape
anomaly_heatmap_reshaped = np.reshape(anomaly_heatmap, (anomalous_generator.samples, img_height, img_width))
print(anomaly_heatmap.shape)
print(anomalous_generator[0][0].shape)

# Plot the first anomalous image and its corresponding heatmap
plt.subplot(1, 2, 1)
plt.imshow(anomalous_generator[0][0])
plt.title('Anomalous Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(anomaly_heatmap_reshaped[0], cmap='hot')
plt.title('Anomaly Heatmap')
plt.axis('off')

plt.show()



model.save('OCC.h5')

occ_model=tf.keras.models.load_model('OCC.h5')
occ_model.summary()

import cv2

img=cv2.imread(r"G:\Professor_Anomaly_detection\Final_code_and_results\Final_code_Materials\One_class_classification\Input\Normal\N2 (103).jpg")

#cv2.imshow('img',img)
 
print(img.shape)

# img=cv2.resize(img,(256,256))
# img=img.reshape(1,256,256,3)

# anomaly_scores = model.predict(img)

# anomaly_heatmap = tf.reduce_mean(tf.square(anomaly_scores - 0.5), axis=-1)

import os 
import scipy
import numpy as np


occ_model.layers[-4].get_weights()[0]


inputFramesPath = r'Input\Abnormal/'
inputdata = os.listdir(inputFramesPath)

for frame in inputdata:
    fullPathImage = os.path.join(inputFramesPath, frame)
    testImage = cv2.imread(fullPathImage)
    dim = (256, 256)
    testImage = cv2.resize(testImage, dim)
    testImage = testImage / 255.
    
    anomaly_scores = occ_model.predict(np.expand_dims(testImage, axis=0))
    anomaly_heatmap = tf.reduce_mean(tf.square(anomaly_scores - 0.5), axis=-1)
    anomaly_scores = np.argmax(anomaly_scores)
    
    last_layer_weights = occ_model.layers[-4].get_weights()[0] 
    last_layer_weights_for_pred = last_layer_weights[:, anomaly_scores]
    last_conv_model = tf.keras.Model(occ_model.input, occ_model.get_layer("conv2d_2").output)
    last_conv_output = last_conv_model.predict(testImage[np.newaxis,:,:,:])
    last_conv_output = np.squeeze(last_conv_output)
    h = int(testImage.shape[0] / last_conv_output.shape[0])
    w = int(testImage.shape[1] / last_conv_output.shape[1])
    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
    
    heat_map = np.dot(upsampled_last_conv_output.reshape((testImage.shape[0]*testImage.shape[1], 128)), 
                      last_layer_weights_for_pred)
    heat_map = heat_map.reshape(testImage.shape[0], testImage.shape[1])
    heat_map[testImage[:, :, 0] == 0] = 0  
    
    peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=10)
  
    plt.imshow(testImage.astype('float32').reshape(testImage.shape[0], testImage.shape[1], 3))
    plt.imshow(heat_map, cmap='jet', alpha=0.30)
    plt.axis('off')










































# ast_conv_model = tf.keras.Model(occ_model.input, model.get_layer("conv2d_5").output)

inputFramesPath=r'Input\Abnormal/' #abnormal01
# inputFramesPath=r'Input\SelectedFrames\TrainGridCame\Normal/'
inputdata=os.listdir(inputFramesPath)
#print(inputdata)
for frame in inputdata:
    fullPathImage=os.path.join(inputFramesPath,frame)
    testImage=cv2.imread(fullPathImage)
    dim = (256, 256)
    testImage=cv2.resize(testImage, dim)
    testImage=testImage /255.
    ###########
    anomaly_scores = occ_model.predict(np.expand_dims(testImage, axis=0))
    anomaly_heatmap = tf.reduce_mean(tf.square(anomaly_scores - 0.5), axis=-1)
    #print(anomaly_heatmap)
    anomaly_scores = np.argmax(anomaly_scores)
    #print(pred_class)
    print(anomaly_heatmap)

    last_layer_weights = occ_model.layers[-4].get_weights()[0] 
    last_layer_weights_for_pred = last_layer_weights[:, anomaly_scores]
    last_layer_weights_for_pred = np.expand_dims(last_layer_weights_for_pred, axis=(0, 1))
    last_layer_weights_for_pred = np.tile(last_layer_weights_for_pred, (testImage.shape[0], testImage.shape[1], 1))
    
    last_conv_model = tf.keras.Model(occ_model.input, occ_model.get_layer("conv2d_2").output)
    last_conv_output = last_conv_model.predict(testImage[np.newaxis,:,:,:])
    last_conv_output = np.squeeze(last_conv_output)
    h = int(testImage.shape[0]/last_conv_output.shape[0])
    w = int(testImage.shape[1]/last_conv_output.shape[1])
    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
    
    print('upsampled_last_conv_output',upsampled_last_conv_output.shape)
    heat_map = np.dot(upsampled_last_conv_output.reshape((testImage.shape[0]*testImage.shape[1], 3, 128)), 
                      last_layer_weights_for_pred.reshape((1, 1, 128))).reshape(testImage.shape[0],testImage.shape[1])
    heat_map[testImage[:,:,0] == 0] = 0  
    peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=10)
  
    plt.imshow(testImage.astype('float32').reshape(testImage.shape[0],testImage.shape[1],3))
    plt.imshow(heat_map, cmap='jet', alpha=0.30)
    plt.axis('off')



















# # Normalize the heatmap to be between 0 and 1
anomaly_heatmap = (anomaly_heatmap - tf.reduce_min(anomaly_heatmap)) / (tf.reduce_max(anomaly_heatmap) - tf.reduce_min(anomaly_heatmap))

# Visualize the heatmap for the anomalous images
import matplotlib.pyplot as plt
print(anomaly_heatmap.shape)
plt.imshow(anomaly_heatmap, cmap='hot')
# img = img.swapaxes(0,1)
# img = img.swapaxes(1,2)
plt.show()
# # Normalize the heatmap to be between 0 and 1

# anomaly_heatmap = (anomaly_heatmap - tf.reduce_min(anomaly_heatmap)) / (tf.reduce_max(anomaly_heatmap) - tf.reduce_min(anomaly_heatmap))

# # Reshape the heatmap to have dimensions (height, width, batch_size, 1)
# anomaly_heatmap = tf.expand_dims(anomaly_heatmap, axis=-1)

# # Transpose the heatmap to have dimensions (height, width, batch_size, 1)
# anomaly_heatmap = tf.transpose(anomaly_heatmap, perm=[1, 2, 0, 3])
# anomaly_heatmap = tf.squeeze(anomaly_heatmap, axis=-1)

# # Visualize the heatmap for the anomalous images
# import matplotlib.pyplot as plt
# if anomaly_heatmap.ndim == 4:
#     plt.imshow(anomaly_heatmap[:,:,0,0], cmap='hot')
#     plt.show()
# else:
#     print("Invalid heatmap shape for visualization.")