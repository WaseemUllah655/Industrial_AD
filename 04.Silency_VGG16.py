import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam, SGD
import scipy  
import os
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras import layers 
from matplotlib.patches import Rectangle 
from skimage.feature.peak import peak_local_max  
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
import tensorflow as tf
import time

SIZE = 224

saveResults=r'Results\04.Silency_vgg16/'

trainedModel=r'Trained_Model\model_VGG16.h5'
model=tf.keras.models.load_model(trainedModel)

inputFramesPath=r'SelectedFrames\abnormal01/' #abnormal01
#inputFramesPath=r'inputData\SelectedFrames\noarml01/'
inputdata=os.listdir(inputFramesPath)
num_frames = 95;
start = time.time()

##############################################
def veiw_saliency_map(imge, the_class,model):
    #The image needs to be preprocessed before being fed to the model
    # read the image
    img =cv2.imread(imge)
    # resize to 300x300 and normalize pixel values to be in the range [0, 1]
    img = cv2.resize(img, (224, 224)) / 255.0
    # add a batch dimension in front
    image = np.expand_dims(img, axis=0)
    #------------------------------------------------------------------------
    # compute the gradient of the loss with respect to the input image,
    # this should tell us how the loss value changes with respect to
    # a small change in input image pixels
    expected_output = tf.constant(the_class, shape=(1, 2))
    classProbabilites= np.argmax(model(image))
    if classProbabilites==0:
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        #plt.show() 
        plt.savefig(saveResults+'01.Normal/'+str(frame)+'.jpg')
    if classProbabilites==1:
        print('classProbabilites',classProbabilites)
        with tf.GradientTape() as tape:
            # cast image to float
            inputs = tf.cast(image, tf.float32)
            # watch the input pixels
            tape.watch(inputs)
            # generate the predictions
            predictions = model(inputs)
            # get the loss
            loss = tf.keras.losses.binary_crossentropy(expected_output, predictions)
    
        # get the gradient with respect to the inputs
        gradients = tape.gradient(loss, inputs)
        # The gradient tells us how much the categorical loss changes when the input change,
        #also it can indicate when certain parts of the input image have a
        #significat effect on the categorical prediction
        #------------------------------------------------------------------------
        # reduce the RGB image to grayscale
        grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1)
    
        # normalize the pixel values to be in the range [0, 255].
        # the max value in the grayscale tensor will be pushed to 255.
        # the min value will be pushed to 0.
        normalized_tensor = tf.cast(255
            * (grayscale_tensor - tf.reduce_min(grayscale_tensor))
            / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)),
            tf.uint8,)
        # remove the channel dimension to make the tensor a 2d tensor
        normalized_tensor = tf.squeeze(normalized_tensor)
        #------------------------------------------------------------------------
        gradient_color = cv2.applyColorMap(normalized_tensor.numpy(), cv2.COLORMAP_HOT)
        gradient_color = gradient_color / 255.0
        super_imposed = cv2.addWeighted(img, 0.5, gradient_color, 0.5, 0.0)
        plt.figure(figsize=(8, 8))
        plt.imshow(super_imposed)
        plt.axis('off')
        #plt.show()    
        plt.savefig(saveResults+'02.Abnormal/'+str(frame)+'.jpg')
    
##############################################
for frame in inputdata:
    fullPathImage=os.path.join(inputFramesPath,frame)
    veiw_saliency_map(fullPathImage,0,model)#imge, the_class,model

