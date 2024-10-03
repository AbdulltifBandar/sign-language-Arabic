import tensorflow as tf
from keras import preprocessing
import os
import keras
from keras import Sequential
from keras import layers
from keras.src.legacy.preprocessing.image import ImageDataGenerator


dataset_dir="ArASL_Database_54K_Final"


#incres data number
datagen =ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='constant',
    validation_split=0.4
)


#split data into training and validation

training_ds=datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    color_mode = 'grayscale',
    subset = 'training',  # Subset for training
    seed = 123)

validation_ds=datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    color_mode = 'grayscale',
    subset = 'validation',  # Subset for training
    seed = 123)




# make data values between 0 and 1
#training=training_ds.map(lambda x ,y:(x/255,y))
#validation=validation_ds.map(lambda x ,y:(x/255,y))


#build neural network
model=Sequential()

# First Convolutional Layer
model.add(layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', input_shape=(64, 64, 1)))
model.add(layers.BatchNormalization())  # Added Batch Normalization for stability
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(256, (3, 3), strides=1, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Dense(32, activation='softmax'))



#start training
model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(training_ds,validation_data=validation_ds,epochs=100)

#save model 
model.save('my_model_2.h5')