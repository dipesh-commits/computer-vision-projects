import cv2
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator


train_data_path = 'hand_gesture/dataset/training_set'
test_data_path = 'hand_gesture/dataset/test_set'

batch_size=32


train_datagen = ImageDataGenerator(rotation_range=40,rescale=1./255,shear_range=0.2,horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)



model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (70,90,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation = "softmax"))


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])




train_generator = train_datagen.flow_from_directory(train_data_path,target_size=(70,90),batch_size=batch_size,class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(test_data_path,target_size=(70,90),batch_size=batch_size,class_mode='categorical')

History = model.fit_generator(train_generator,steps_per_epoch=2000//batch_size,epochs=20,validation_data=valid_generator,validation_steps=800//batch_size)

model.save('hand_model.h5')