# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:10:55 2023

@author: Waseem_Ullah
"""

import numpy as np
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
        './input/',
        target_size=(256, 256),
        batch_size=32,
        class_mode='input')

input_img = Input(shape=(256, 256, 3))
x = Flatten()(input_img)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
encoded = Dense(32, activation='relu')(x)
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
decoded = Dense(256*256*3, activation='sigmoid')(x)
decoded = Reshape((256, 256, 3))(decoded)

model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50)

test_img = np.random.rand(1, 256, 256, 3)
test_img = np.expand_dims(test_img, axis=0)
reconstructed_img = model.predict(test_img)
loss = np.mean(np.abs(reconstructed_img - test_img))





# import numpy as np
# from keras.layers import Input, Dense, Flatten, Reshape
# from keras.models import Model
# from keras.preprocessing.image import ImageDataGenerator


# datagen = ImageDataGenerator(rescale=1./255)
# train_generator = datagen.flow_from_directory(
#         './input/',
#         target_size=(256, 256),
#         batch_size=32,
#         class_mode='input')



# input_img = Input(shape=(256, 256, 3))
# x = Flatten()(input_img)
# x = Dense(256, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# encoded = Dense(32, activation='relu')(x)
# x = Dense(64, activation='relu')(encoded)
# x = Dense(128, activation='relu')(x)
# x = Dense(256, activation='relu')(x)
# decoded = Dense(256*256*3, activation='sigmoid')(x)
# decoded = Reshape((256, 256, 3))(decoded)

# model = Model(input_img, decoded)
# model.compile(optimizer='adam', loss='binary_crossentropy')

# model.fit_generator(
#         train_generator,
#         steps_per_epoch=2000,
#         epochs=50)

# test_img = np.random.rand(1, 256, 256, 3)
# test_img = np.expand_dims(test_img, axis=0)
# reconstructed_img = model.predict(test_img)
# loss = np.mean(np.abs(reconstructed_img - test_img))