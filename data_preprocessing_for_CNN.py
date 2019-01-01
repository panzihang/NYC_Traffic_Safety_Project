# This file first categorizes the number of accidents into 5 categories (0,1,2,3,4) as the labels.
# Then it combines the labels with the images by coordinates. Save the combined file as a pickle file for training use.

# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore') # Filter out warnings

import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2

# for copying data
import copy


# In[2]:


# Read accident data
accidents = pd.read_csv('accidents_count.csv')

# Split into zero and non-zero parts
acci_zero = accidents[accidents['acci_counts']==0]
acci_nonzero = accidents[accidents['acci_counts']!=0]


# In[3]:


# Label non-zero data by numbers from 1 to 4
# 1 denoting relatively safe, 2 denoting medium, 3 denoting realtively dangerous, and 4 denoting dangerous
acci_nonzero['label'] = pd.qcut(acci_nonzero['acci_counts'], 4,                         labels=["1", "2", "3", "4"])

# Label zero data by number zero
# 0 denoting safe
acci_zero['label'] = 0

# Randomly sample from the original zero data
safety = acci_zero.sample(n=700)


# In[4]:


# Concatenate the sampled zero data (safety) and the nonzero data
combine = pd.concat([acci_nonzero,safety])

# Sort the data by index
combine = combine.sort_index()


# In[5]:


combine.head()


# In[6]:


# Drop unnecessary column (we have labeled accident counts data, so we do not need this column any more)
combine.drop('acci_counts',axis=1,inplace=True)


# In[7]:


combine.head()


# In[8]:


# Create a new dataframe copied from combine
new = copy.deepcopy(combine)

# Transform the former location data to latitude and longitude columns
new['latitude'] = 0
new['longitude'] = 0
for i in range(0,len(new)):
    new.iloc[i,2] = float(new.iloc[i,0].lstrip('(').rstrip(')').split(',')[0])
    new.iloc[i,3] = float(new.iloc[i,0].lstrip('(').rstrip(')').split(',')[1].lstrip())

# Transfrom the former location data to data without blank
for i in range(0,len(new)):
    new.iloc[i,0] = str('('+str(new.iloc[i,2])+',' + str(new.iloc[i,3])+')')


# In[9]:


new.head()


# In[10]:


# Drop the latitude and longitude columns
new.drop('latitude', axis=1, inplace=True)
new.drop('longitude', axis=1, inplace=True)


# In[11]:


# Read intersections data
intersections = pd.read_csv('intersection.csv')
intersections.latitude = round(intersections.latitude,7)
intersections.longitude = round(intersections.longitude,7)
intersections['location']=0


# In[12]:


# Create a location column in the same style with "combine"
for i in range(0,len(intersections)):
    intersections.iloc[i,2] = str('('+str(intersections.iloc[i,0])+',' + str(intersections.iloc[i,1])+')')


# In[13]:


# Create an index column in the intersections dataframe, for the convenience of linking back to image files
intersections['index']=intersections.index


# In[14]:


intersections.head()


# In[15]:


position = copy.deepcopy(intersections)

# Drop the latitude and longitude columns
position.drop('latitude', axis=1, inplace=True)
position.drop('longitude', axis=1, inplace=True)


# In[16]:


#merge tables
position = position.merge(new,on = 'location',how = 'inner')
position.head()


# In[17]:


# Make sure the shape is right
position.shape


# In[18]:


# Add a new column for map data in position
position['image'] = 0
position.head(2)


# In[19]:


# Read images and plug image data into 'image' column

coord_img_dict={}

for i in range(0,len(position)):
    
    # Get the index of path of each map
    idx = position.iloc[i,1]
    img_ = str('intersection_img_size20/'+'testimage_big'+str(idx)+'.png')
    
    # Read map data into a dictionary
    coord_img_dict[i]=cv2.imread(img_)
    
# Plug the dictionary into the position dataframe
maps = pd.DataFrame(list(coord_img_dict.items()), columns=['sequence','image'])
position['image'] = maps['image']


# In[20]:


position.head()


# In[21]:


type(position['image'][0])


# In[22]:


position['image'][0].shape


# In[23]:


import pickle
import os.path

file_path = "position_df.pkl"
# n_bytes = 2**31
max_bytes = 2**31 - 1
# data = bytearray(n_bytes)

## write
bytes_out = pickle.dumps(position)
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])


# In[24]:


## read
bytes_in = bytearray(0)
input_size = os.path.getsize(file_path)
with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
data2 = pickle.loads(bytes_in)

