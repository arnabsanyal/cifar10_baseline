# Import all modules
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras import backend as K
import h5py

if K.backend()=='tensorflow':
    K.set_image_dim_ordering('th')
 
# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp
 
# Loading the CIFAR-10 datasets
cifar10 = h5py.File('cifar10.hdf5', 'r')

# Declare variables

batch_size = 50 
# 50 examples in a mini-batch, smaller batch size means more updates in one epoch

num_classes = 10 #
epochs = 100 # repeat 100 times

x_train = np.array(cifar10['train_data']) / 255.0
x_test = np.array(cifar10['test_data']) / 255.0
y_train = np.array(cifar10['train_labels'])
y_test = np.array(cifar10['test_labels'])

x_train.shape = 50000, 3, 32, 32
x_test.shape = 10000, 3, 32, 32

def base_model_four_layers():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Train model

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

cnn_n = base_model_four_layers()
cnn_n.summary()

# Fit model

cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test), shuffle=True)
sequential_model_to_ascii_printout(cnn_n)
cnn_n.save('cifar10_4c2f_model.h5')

# Plots for training and testing process: loss and accuracy

fig1 = plt.figure(figsize = (16, 9))
ax1 = fig1.add_subplot(111)
ax1.plot(cnn.history['acc'],'r')
ax1.plot(cnn.history['val_acc'],'g')
plt.xticks(np.arange(0, 101, 2.0))
ax1.set_xlabel("Num of Epochs")
ax1.set_ylabel("Accuracy")
ax1.set_title("Training Accuracy vs Validation Accuracy")
ax1.legend(['train','validation'])
plt.grid(which='both', axis='both', linestyle='-.')
plt.savefig('accuracy.png')


fig2 = plt.figure(figsize = (16, 9))
ax2 = fig2.add_subplot(111)
ax2.plot(cnn.history['loss'],'r')
ax2.plot(cnn.history['val_loss'],'g')
plt.xticks(np.arange(0, 101, 2.0))
ax2.set_xlabel("Num of Epochs")
ax2.set_ylabel("Loss")
ax2.set_title("Training Loss vs Validation Loss")
ax2.legend(['train','validation'])
plt.grid(which='both', axis='both', linestyle='-.')
plt.savefig('loss.png')

# plt.show()

scores = cnn_n.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))