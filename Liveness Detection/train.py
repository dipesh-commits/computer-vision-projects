import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False






img_height = 360
img_width = 360
input_shape = (360,360,3)
num_classes = 2
batch_size = 8
epochs = 30
lr = 0.001

datapath = 'dataset'   #input data path

augmentor = ImageDataGenerator(rescale= 1./255,rotation_range=20, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,horizontal_flip=True, fill_mode="nearest",validation_split=0.2)


train_generator = augmentor.flow_from_directory(datapath,target_size=(img_height, img_width),batch_size=batch_size,class_mode='categorical', subset='training') # set as training data

validation_generator = augmentor.flow_from_directory(datapath,target_size=(img_height, img_width),batch_size=batch_size, class_mode='categorical',subset='validation') # set as validation data



model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu',input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

print("Compiling model.....")

model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=lr, decay=lr / epochs),metrics=["acc"])
print("Compiling finished....")

print("Model Training for {} epochs".format(epochs))

History = model.fit(train_generator,steps_per_epoch = train_generator.samples // batch_size,validation_data = validation_generator, validation_steps = validation_generator.samples // batch_size, epochs=epochs)
print("Training finished....")

print("Saving model.....")
model.save('model.h5')


print("Printing graph...")
plt.plot(History.history['acc'])
plt.xlabel(['Epochcount'])
plt.plot(History.history['val_acc'])
plt.ylabel(['Accuracy'])
plt.savefig('accuracy.png')
plt.show()



plt.plot(History.history['loss'])
plt.xlabel(['Epochcount'])
plt.plot(History.history['val_loss'])
plt.ylabel(['Lossdata'])
plt.savefig('loss.png')
plt.show()






