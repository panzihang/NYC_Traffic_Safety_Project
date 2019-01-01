# This file reads the accident data and generates the coordinates of all the accidents in Manhattan

# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[71]:


data = pd.read_csv('C:/Users/panyg/Documents/UCB_course/290/project/nyc data/NYPD_Motor_Vehicle_Collisions.csv')


# In[72]:


data.head()


# In[73]:


data = data[data['BOROUGH'] == 'MANHATTAN']


# In[89]:


location = data.loc[:,['LATITUDE','LONGITUDE']]


# In[90]:


location.columns = ['lat','lon']


# In[91]:


location.shape


# In[94]:


location = location[location['lon']<-73.90]
location = location[location['lon']>-74.03]
location = location[location['lat']>40.7]
location = location[location['lat']<40.9]


# In[95]:


plt.scatter(location['lon'],location['lat'])


# In[96]:


right_loc = location.copy().reset_index(drop = True)
right_loc.shape

