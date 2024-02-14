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

savePath=r'Results\03.Canny/02.Normal_vs_Normal/'

normal_Image=r'SelectedFrames\noarml01/N1 (1).jpg'
normal_Image = cv2.imread(normal_Image)

#inputFramesPath=r'inputData\SelectedFrames\abnormal01/' #abnormal01
inputFramesPath=r'SelectedFrames\noarml01/'

#// make kernel matrix for dilation and erosion (Use Numpy)
#kernel = np.ones((2, 2), np.uint8)

inputdata=os.listdir(inputFramesPath)
for frame in inputdata:
    fullPathImage=os.path.join(inputFramesPath,frame)
    print(fullPathImage)
    query_Image = cv2.imread(fullPathImage)
    query_Image_Copy=query_Image
    
    # Convert the images to grayscale
    gray_Normal_Image = cv2.cvtColor(normal_Image, cv2.COLOR_BGR2GRAY)
    query_Image = cv2.cvtColor(query_Image, cv2.COLOR_BGR2GRAY)
    
    # Perform edge detection on both images using structured forest
    Normal_Imageedges = cv2.Canny(gray_Normal_Image, 300, 1000)
    query_Image_edges = cv2.Canny(query_Image, 300, 1000)
    
    # Subtract the edges of one image from another
    diff_Edges = Normal_Imageedges-query_Image_edges
    # Create a mask by thresholding the difference
    diff_Mask = np.where(diff_Edges > 0.5, 10000, 0)
    
    # Convert the mask to red color
    mask = np.dstack([diff_Mask, np.zeros_like(diff_Mask), np.zeros_like(diff_Mask)])
    mask = (mask * 255).astype("uint8")
    
    #mask = cv2.dilate(mask, kernel, iterations=1)  #// make dilation image

    # Overlay the mask on the original image
    #result = query_Image_Copy * mask
    
    result = cv2.addWeighted(query_Image_Copy,0.5,mask,0.7,0)
    # Plot the original image and the result
    fig, (ax0, ax1,ax2) = plt.subplots(nrows=1, ncols=3, figsize=(16,8),sharex=True, sharey=True)
    ax0.imshow(normal_Image)
    ax0.set_title('Normal Image')
    ax1.imshow(query_Image_Copy)
    ax1.set_title('Abnormal Image')
    ax2.imshow(result)
    ax2.set_title('Abnormal Region')
    plt.tight_layout()
    plt.show()
    plt.savefig(savePath+str(frame)+'.jpg')
    
    
    
        
    
    

