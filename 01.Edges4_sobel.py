# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 01:28:55 2023

@author: IMLAB_614
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, color
import cv2 

# Load the images
image1 = cv2.imread(r"G:\Professor_Anomaly_detection\Data_and_Code\Data\Clear_Normal\noarml01.mp4_frame273.jpg")
image2 = cv2.imread(r"G:\Professor_Anomaly_detection\Data_and_Code\Data\Clear_Abnormal\abnormal01.mp4_frame284.jpg")


# Convert the images to grayscale
gray_image1 = color.rgb2gray(image1)
gray_image2 = color.rgb2gray(image2)

# Perform edge detection on both images using structured forest
edges1 = filters.sobel(gray_image1)
edges2 = filters.sobel(gray_image2)

# Subtract the edges of one image from another
diff_edges = edges1 - edges2

# Create a mask by thresholding the difference
mask = np.where(diff_edges > 0.4, 1000000, 0)


# Convert the mask to red color
mask = np.dstack([mask, np.zeros_like(mask), np.zeros_like(mask)])

# Overlay the mask on the original image
result = image1 * mask

# Plot the original image and the result
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),sharex=True, sharey=True)

ax0.imshow(image1)
ax0.set_title('Original Image')

ax1.imshow(result)
ax1.set_title('Masked Image')

plt.tight_layout()
plt.show()




