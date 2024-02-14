# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
import anogan
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.astype(np.float32)/255.
# X_train = X_train.reshape(60000, 28, 28, 1)

saveFolder=r"F:\IMLAB_WORK\Projects\02.Manifactures_Anomlies\Results\07.AnoGAN/"
image_directory = 'SelectedFrames/TrainGridCame/'

epoches=30
SIZE = 28


dataset1 = [] 
labe1l = [] 


Clear_Abnormal = os.listdir(image_directory + 'Abnormal/')
#print('Clear_Abnormal =>',Clear_Abnormal)
for i, image_name in enumerate(Clear_Abnormal):   
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'Abnormal/' + image_name,0)
        #image = Image.fromarray(image, 'RGB')
        #image = image.resize((SIZE, SIZE))
        image=cv2.resize(image,(SIZE,SIZE))
        dataset1.append(np.array(image))
        labe1l.append(1)

dataset = [] 
label = []  

Clear_Normal_images = os.listdir(image_directory + 'Normal/')
for i, image_name in enumerate(Clear_Normal_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'Normal/' + image_name,0)
        #image = Image.fromarray(image, 'RGB')
        #image = image.resize((SIZE, SIZE))
        image=cv2.resize(image,(SIZE,SIZE))
        dataset.append(np.array(image))
        label.append(0)


dataset = np.array(dataset)
label = np.array(label)

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)

X_train = X_train /255.
X_test = X_test /255.


##################
dataset1 = np.array(dataset1)
label1 = np.array(dataset1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset1, label1, test_size = 0.20, random_state = 0)

X_train1 = X_train1 /255.
X_test1 = X_test1 /255.

#########################################
plt.figure(figsize=(2, 2))
plt.imshow(X_train1[2].reshape(28, 28),cmap=plt.cm.gray)
plt.show()

Model_d, Model_g = anogan.train(128, X_train)

#####################
X_test1 = X_test1.astype(np.float32)/255.
X_test1 = X_test.reshape(-1, 28, 28, 1)



for i in range (95):
    print(i)
    image=X_test1[i]
    generated_img = anogan.generate(1)
    #plt.figure(figsize=(2, 2))
    #plt.imshow(generated_img.reshape(28,28), cmap=plt.cm.gray)
    #plt.savefig(saveFolder+str(i)+'.generated_img.jpg')
    ###############
    test_img = image
    plt.figure(figsize=(2, 2))
    plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
    plt.savefig(saveFolder+str(i)+'.input.jpg')
    #############
    model = anogan.anomaly_detector()
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1))
    print("anomaly score : " + str(ano_score))

    plt.figure(figsize=(2, 2))
    plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
    residual  = test_img.reshape(28,28) - similar_img.reshape(28, 28)
    plt.imshow(residual, cmap='jet', alpha=.5)
    plt.savefig(saveFolder+str(i)+'.output.jpg')
    
    plt.show()
        






# ## generate random image 
# generated_img = anogan.generate(13)
# plt.figure(figsize=(2, 2))
# plt.imshow(generated_img[4].reshape(28, 28),cmap=plt.cm.gray)
# plt.savefig(saveFolder+'1.generated.jpg')
# plt.show()


# ## compute anomaly score - sample from test set
# X_test = X_test1.astype(np.float32)/255.
# X_test = X_test.reshape(-1, 28, 28, 1)
# test_img = X_test[0]

# model = anogan.anomaly_detector()
# ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1))

# plt.figure(figsize=(2, 2))
# plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
# plt.savefig(saveFolder+'1.orignal.jpg')
# plt.show()

# print("anomaly score : " + str(ano_score))

# plt.figure(figsize=(2, 2))
# plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
# residual  = test_img.reshape(28,28) - similar_img.reshape(28, 28)
# plt.imshow(residual, cmap='jet', alpha=.5)
# plt.savefig(saveFolder+'1.output.jpg')
# plt.show()
















# ## compute anomaly score - sample from strange image
# test_img = plt.imread('assets/test_img.png')
# test_img = test_img[:,:,0]

# model = anogan.anomaly_detector()
# ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1))

# plt.figure(figsize=(2, 2))
# plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
# plt.show()

# print("anomaly score : " + str(ano_score))
# plt.figure(figsize=(2, 2))
# plt.imshow(test_img.reshape(28,28), cmap=plt.cm.gray)
# residual  = test_img.reshape(28,28) - similar_img.reshape(28, 28)
# plt.imshow(residual, cmap='jet', alpha=.5)
# plt.show()


# from sklearn.manifold import TSNE
# ## t-SNE embedding 
# # generating anomaly image for test (radom noise image)

# random_image = np.random.uniform(0,1, (100, 28,28, 1))
# print("a sample from generated anomaly images(random noise image)")
# plt.figure(figsize=(2, 2))
# plt.imshow(random_image[0].reshape(28,28), cmap=plt.cm.gray)
# plt.show()

# # intermidieate output of discriminator
# model = anogan.feature_extractor()
# feature_map_of_random = model.predict(random_image, verbose=1)
# feature_map_of_minist = model.predict(X_test[:300], verbose=1)

# # t-SNE for visulization
# output = np.concatenate((feature_map_of_random, feature_map_of_minist))
# output = output.reshape(output.shape[0], -1)
# anomaly_flag = np.array([1]*100+ [0]*300)

# X_embedded = TSNE(n_components=2).fit_transform(output)
# plt.title("t-SNE embedding on the feature representation")
# plt.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
# plt.scatter(X_embedded[100:,0], X_embedded[100:,1], label='minist(normal)')
# plt.legend()
# plt.show()