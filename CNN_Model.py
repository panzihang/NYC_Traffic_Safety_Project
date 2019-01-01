# This file still contains a little preprocessing, including one-hot-encoding, and normalization.
# It then built a CNN Model, and trains the model with images and the categorized labels.
# It returns the performance of the model. And we will do further modifications on the model to improve performance.


# coding: utf-8

# ## 1. Package Preparation

# In[1]:


#import packages

# clear warnings
import warnings
warnings.simplefilter("ignore")

# import keras data
from keras import backend as K
K.set_image_dim_ordering('tf') # note that we need to have tensorflow dimension ordering still because of the weigths.
print('The backend is:',K.backend())
import tensorflow as tf
print(K.image_dim_ordering()) # should say tf
print(tf.__version__) # tested for 1.11.0
import cv2
import keras
print(keras.__version__) # tested for 2.2.4


# In[2]:


# Import relevant packages
from __future__ import absolute_import, division, print_function # make it compatible w Python 2
import os
import h5py # to handle weights
import os, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# to read image
from PIL import Image

# relative keras packages
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, ZeroPadding2D
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import model_from_json
from keras.preprocessing import image

# useful packages from sklearn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from IPython.display import Image, display

# fix random seed for reproducibility
np.random.seed(150)


# In[3]:


import cv2


# ## 2. Read data from preprocessed pickle file

# In[4]:


# We have preprocessed our data and saved in a pickle file
# Now we only need to read this pickle file
map_data = pd.read_pickle('position_df.pkl')


# In[5]:


## RESIZE

map_data['image'] = map_data['image'].apply(lambda x: cv2.resize(x,(299,299)))


# In[6]:


map_data.iloc[0,1].shape


# The column 'index' in the map_data above is used to help match the location data for the 'location' column.
# 
# We only need the data in the 'label' column and 'image' column to train our model, so we drop those two unnecessary columns.

# In[7]:


map_data.drop(['location','index'],axis=1,inplace=True)


# In[8]:


map_data.shape


# In[9]:


# Show the head of our map_data
map_data.head()


# In[10]:


## Subset

map_data_sub = map_data.iloc[0:600,:]
map_data_sub.shape


# ## 3. Split data for cross validation

# In[11]:


Xs = pd.DataFrame(map_data_sub.iloc[:,1])
ys = pd.DataFrame(map_data_sub.iloc[:,0])
ys['label'] = ys['label'].apply(lambda x: int(x))


# In[12]:


# Split data into training data (including training and validation) and test data
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size = 0.2, random_state = 0)


# In[13]:


# Split training data into true training data and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)


# In[14]:


# confirm the shape and type of our data is right
print('The shape of the first item of X_train is', X_train.iloc[0,0].shape)
print('The length of X_train is', len(X_train))

print('The shape of the first item of X_val is', X_val.iloc[0,0].shape)
print('The length of X_val is', len(X_val))

print('The shape of the first item of X_test is', X_test.iloc[0,0].shape)
print('The length of X_test is', len(X_test))

print('The type and value of the first item of y_train is', type(y_train.iloc[0,0]), y_train.iloc[0,0])
print('The length of y_train is', len(y_train))

print('The shape of the first item of y_val is', type(y_val.iloc[0,0]), y_val.iloc[0,0])
print('The length of y_val is', len(y_val))

print('The type and value of the first item of y_test is', type(y_test.iloc[0,0]), y_test.iloc[0,0])
print('The length of y_test is', len(y_test))


# In[15]:


# A small check of total data amount before training the model
check_X = (len(X_train) + len(X_val) + len(X_test)) == len(map_data_sub)
check_y = (len(y_train) + len(y_val) + len(y_test)) == len(map_data_sub)

print('X data sets equality is:', check_X)
print('y data sets equality is:', check_y)


# ## 4. One-hot encoding and normalizing our input data

# In[16]:


'''# Create combined lists for training and test sets,
# so that we can do normalization and encoding in one function on both data sets.
# Note that 'combine' is only a pointer,
# so when we change something on data sets in combine,
# the original data sets will also change.

combine_X = [X_train, X_val, X_test]
combine_y = [y_train, y_val, y_test]'''


# In[17]:


'''combine_y = [np.array(y_train['label']), np.array(y_val['label']), np.array(y_test['label'])]'''


# In[18]:


'''# normalize inputs from 0-255 to 0.0-1.0
combine_X = combine_X / 255'''


# In[19]:


'''# check the value of normalized data
print('The shape and value of X data after normalization:', df.iloc[0,0].shape, '\n', df.iloc[0,0][0])'''


# In[20]:


## Normalize X

X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255


# In[21]:


#### from sklearn.preprocessing imposrt LabelEncoder, OneHotEncoder


# In[22]:


'''#### one hot encoding our categorical data
#labelencoder_y = LabelEncoder()
#enc = OneHotEncoder()
for df in combine_y:
    df = np_utils.to_categorical(df)
    #df.iloc[:,0] = labelencoder_y.fit_transform(df.iloc[:,0])
    #df = enc.fit(np.array(df).reshape(-1,1))'''


# In[23]:


## One-hot y

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)


# In[24]:


'''# Check the type of y after one-hot encoding
type(y_train.iloc[0,0])'''


# In[25]:


type(y_train)


# In[26]:


len(y_train)


# In[27]:


X_train.shape


# ## 5. Model Training

# The function below is adapted from: 
# http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/

# In[28]:


# First type of CNN model
input_size=(299,299,3)
num_classes=5

def createCNNModel(num_classes):

    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_size, border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    epochs = 3  # >>> should be 25+
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model, epochs

# create our CNN model
model, epochs = createCNNModel(num_classes)
print("CNN Model created.")


# The function below is adapted from: 
# http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/

# In[29]:


'''# Another type of CNN model
input_size=(640,640,3)
num_classes=6

def createCNNModel(num_classes):

    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_size, border_mode='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    epochs = 3  # >>> should be 25+
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model, epochs

# create our CNN model
model, epochs = createCNNModel(num_classes)
print("CNN Model created.")'''


# In[30]:


X_train.shape


# In[31]:


X_train_list = X_train['image'].tolist()


# In[32]:


X_train_array = np.array(X_train_list)
y_train_array = np.array(y_train)


# In[33]:


X_train_array


# In[34]:


len(X_train_array)


# In[35]:


y_train_array


# In[36]:


#batch size，learning rate can be modified before training

batch_size=60
seed = 7
np.random.seed(seed)
model.fit(X_train_array, y_train_array, batch_size = batch_size, nb_epoch = epochs)
#model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=60)


# In[42]:


X_test_array = np.array(X_test['image'].tolist())
y_test_array = np.array(y_test)
X_val_array =np.array(X_val['image'].tolist())
y_val_array = np.array(y_val)


# In[43]:


scores = model.evaluate(X_train_array, y_train_array, verbose=0)
print("Train Accuracy: %.2f%%" % (scores[1]*100))

scores = model.evaluate(X_val_array, y_val_array, verbose=0)
print("val Accuracy: %.2f%%" % (scores[1]*100))

# Final evaluation of the model
scores = model.evaluate(X_test_array, y_test_array, verbose=0)
print("Test Accuracy: %.2f%%" % (scores[1]*100))

print("done")
