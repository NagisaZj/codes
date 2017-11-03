import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import display ,Image,HTML
import cv2
import tensorflow as tf
import pandas as pd
import keras as ks
from keras.layers import core,Dense,Conv2DTranspose,Conv2D,LeakyReLU,convolutional
from keras.layers.normalization import BatchNormalization as b_n
from keras.optimizers import Adam
from keras.models import Sequential
IMAGE_SIZE=150
CHANNELS=3
pixel_depth=255.0

TRAIN_DIR = 'input/train/'
TEST_DIR = 'input/test/'

OUTFILE='output.npsave.bin'
TRAINING_AND_VALIDATION_SIZE_DOGS = 100
TRAINING_AND_VALIDATION_SIZE_CATS = 100
TRAINING_AND_VALIDATION_SIZE_ALL = 200
TRAINING_SIZE = 160
VALID_SIZE = 40
TEST_SIZE_ALL = 50

if (TRAINING_SIZE + VALID_SIZE != TRAINING_AND_VALIDATION_SIZE_ALL):
   print ("Error, check that TRAINING_SIZE+VALID_SIZE is equal to TRAINING_AND_VALIDATION_SIZE_ALL")
   exit ()

train_images=[TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

train_images = train_dogs[:TRAINING_AND_VALIDATION_SIZE_DOGS] + train_cats[:TRAINING_AND_VALIDATION_SIZE_CATS]
train_labels = np.array ((['dogs'] * TRAINING_AND_VALIDATION_SIZE_DOGS) + (['cats'] * TRAINING_AND_VALIDATION_SIZE_CATS))
test_images =  test_images[:TEST_SIZE_ALL]
test_labels = np.array (['unknownclass'] * TEST_SIZE_ALL)

def read_image(file_path):
    img=cv2.imread(file_path,cv2.IMREAD_COLOR)
    if (img.shape[0] >= img.shape[1]):  # height is greater than width
        resizeto = ( int(round(IMAGE_SIZE * (float(img.shape[1]) / img.shape[0]))),IMAGE_SIZE);
    else:
        resizeto = ( IMAGE_SIZE,int(round(IMAGE_SIZE * (float(img.shape[0]) / img.shape[1]))));

    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation=cv2.INTER_CUBIC)
    img3 = cv2.copyMakeBorder(img2, 0, IMAGE_SIZE - img2.shape[0], 0, IMAGE_SIZE - img2.shape[1], cv2.BORDER_CONSTANT,
                              0)

    return img3[:, :, ::-1]  # turn into rgb format


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)

    for i, image_file in enumerate(images):
        image = read_image(image_file);
        image_data = np.array(image, dtype=np.float32);
        image_data[:, :, 0] = (image_data[:, :, 0].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:, :, 1] = (image_data[:, :, 1].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:, :, 2] = (image_data[:, :, 2].astype(float) - pixel_depth / 2) / pixel_depth

        data[i] = image_data;  # image_data.T
        if i % 250 == 0: print('Processed {} of {}'.format(i, count))
    return data


train_normalized = prep_data(train_images)
test_normalized = prep_data(test_images)

print("Train shape: {}".format(train_normalized.shape))
print("Test shape: {}".format(test_normalized.shape))
'''
plt.imshow (train_normalized[0,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_normalized[1,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_normalized[2,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_normalized[1000,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_normalized[1001,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_normalized[1002,:,:,:], interpolation='nearest')
#plt.show()
'''
np.random.seed (133)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset_rand, train_labels_rand = randomize(train_normalized, train_labels)
test_dataset, test_labels = randomize(test_normalized, test_labels)

# split up into training + valid
valid_dataset = train_dataset_rand[:VALID_SIZE,:,:,:]
valid_labels =   train_labels_rand[:VALID_SIZE]
train_dataset = train_dataset_rand[VALID_SIZE:VALID_SIZE+TRAINING_SIZE,:,:,:]
train_labels  = train_labels_rand[VALID_SIZE:VALID_SIZE+TRAINING_SIZE]
print ('Training', train_dataset.shape, train_labels.shape)
print ('Validation', valid_dataset.shape, valid_labels.shape)
print ('Test', test_dataset.shape, test_labels.shape)

image_size = IMAGE_SIZE # TODO: redundant, consolidate
num_labels = 2
num_channels = 3 # rg

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (labels=='cats').astype(np.float32); # set dogs to 0 and cats to 1
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print ('Training set', train_dataset.shape, train_labels.shape)
print ('Validation set', valid_dataset.shape, valid_labels.shape)
print ('Test set', test_dataset.shape, test_labels.shape)

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

def model():
    inputs=ks.layers.Input(shape=(150,150,3))
    #conv2d
    x0=ks.layers.Conv2D(128,(3,3),padding='same')(inputs)
    x0=b_n()(x0)
    x00=ks.layers.core.Activation('relu')(x0)
    #residual
    x1=ks.layers.Conv2D(128,(3,3),padding='same')(inputs)
    x2=b_n()(x1)
    x3=ks.layers.core.Activation('relu')(x2)
    x4=ks.layers.Conv2D(128,(3,3),padding='same')(x3)
    x5=b_n()(x4)
    x6=ks.layers.merge([x00,x5],mode='sum')
    x7=ks.layers.core.Activation('relu')(x6)
    #residual end
    # residual
    x1 = ks.layers.Conv2D(128, (3, 3), padding='same')(x7)
    x2 = b_n()(x1)
    x3 = ks.layers.core.Activation('relu')(x2)
    x4 = ks.layers.Conv2D(128, (3, 3), padding='same')(x3)
    x5 = b_n()(x4)
    x6 = ks.layers.merge([x00, x5], mode='sum')
    x7 = ks.layers.core.Activation('relu')(x6)
    # residual end
    # residual
    x1 = ks.layers.Conv2D(128, (3, 3), padding='same')(x7)
    x2 = b_n()(x1)
    x3 = ks.layers.core.Activation('relu')(x2)
    x4 = ks.layers.Conv2D(128, (3, 3), padding='same')(x3)
    x5 = b_n()(x4)
    x6 = ks.layers.merge([x00, x5], mode='sum')
    x7 = ks.layers.core.Activation('relu')(x6)
    # residual end
    flat=core.Flatten()(x7)
    dense1=Dense(1000,activation='relu')(flat)
    out=Dense(2)(dense1)
    model=ks.models.Model(inputs=inputs,outputs=out)
    return model


mod=model()
mod.compile(optimizer='Adam',loss='binary_crossentropy')
def train(batchsize,epochs):
    mod.fit(train_dataset,train_labels,batch_size=batchsize,epochs=epochs,validation_split=0.3)

train(10,1000)