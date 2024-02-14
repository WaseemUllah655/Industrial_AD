# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 01:28:55 2023

@author: IMLAB_614
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, color
import cv2 
import os
from tensorflow.keras.models import Model
import tensorflow as tf
import scipy  
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

savePath=r'Results\05.InverseFeaturesMatching\normal/'

#################
resultsPath=r'Results\02.GridCame\01.VG166/'
model=tf.keras.models.load_model(resultsPath+'model.h5')
#####
def featuresEx(img):
    img=cv2.resize(img, (224,224))
    img=img /255.
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = np.argmax(pred)
    last_layer_weights = model.layers[-1].get_weights()[0] 
    last_layer_weights_for_pred = last_layer_weights[:, pred_class]
    last_conv_model = Model(model.input, model.get_layer("block5_conv3").output)
    last_conv_output = last_conv_model.predict(img[np.newaxis,:,:,:])
    last_conv_output = np.squeeze(last_conv_output)
    h = int(img.shape[0]/last_conv_output.shape[0])
    w = int(img.shape[1]/last_conv_output.shape[1])
    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
    
    heat_map = np.dot(upsampled_last_conv_output.reshape((img.shape[0]*img.shape[1], 512)), 
                  last_layer_weights_for_pred).reshape(img.shape[0],img.shape[1])
    #plt.imshow(heat_map)
    return heat_map
    
normal_Image=r'inputData\SelectedFrames\noarml01/N1 (2).jpg'
normal_Image = cv2.imread(normal_Image)

normal_Image=cv2.resize(normal_Image, (224,224))
# normal_Image=featuresEx(normal_Image)

#inputFramesPath=r'inputData\SelectedFrames\abnormal01/' #abnormal01
inputFramesPath=r'inputData\SelectedFrames\noarml01/'

inputdata=os.listdir(inputFramesPath)
for frame in inputdata:
    fullPathImage=os.path.join(inputFramesPath,frame)
    print(fullPathImage)
    query_Image = cv2.imread(fullPathImage)
    query_Image_Copy=query_Image
    query_Image_Copy=cv2.resize(query_Image_Copy, (224,224))
    
    # Perform edge detection on both images using structured forest
    Normal_Imageedges = featuresEx(normal_Image)
    query_Image_edges = featuresEx(query_Image)
    
    # Subtract the edges of one image from another
    diff_Edges = query_Image_edges-Normal_Imageedges
    
    # # Create a mask by thresholding the difference
    diff_Mask = np.where(diff_Edges > 15, 10000, 0)
    
    # # Convert the mask to red color
    mask = np.dstack([diff_Mask, np.zeros_like(diff_Mask), np.zeros_like(diff_Mask)])
    mask = (mask * 255).astype("uint8")
    
    # #mask = cv2.dilate(mask, kernel, iterations=1)  #// make dilation image

    # # Overlay the mask on the original image
    #result = query_Image_Copy * mask
    
    result = cv2.addWeighted(query_Image_Copy,0.5,mask,0.7,0)
    # # Plot the original image and the result
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(16,8),sharex=True, sharey=True)
    ax0.imshow(query_Image_Copy)
    ax0.set_title('Input Image')
    ax1.imshow(result)
    ax1.set_title('Abnormal Region')
    #plt.show()
    plt.savefig(savePath+str(frame)+'.jpg')
    
    
    
        
    
    

