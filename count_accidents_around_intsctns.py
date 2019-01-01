# This file first extracts all the accident coordinates in Manhattan, then combines them with the coordinates of intersections.
# It returns a csv file providing the coordinate of each intersection 
# and the number of accidents around the intersection (in the range of a satellite image) respectively.


# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore') # Filter out warnings

import xml.etree.ElementTree as ET


# In[2]:


nypd = pd.read_csv("nypd.csv")

# constraint nypd data to manhattan
nypd = nypd[nypd['BOROUGH']=='MANHATTAN']
nypd = nypd[['LATITUDE','LONGITUDE']]


# In[3]:


## delete all rows with 0 values
nypd = nypd.loc[nypd.ne(0).all(axis=1)]
nypd = nypd.dropna()

latitude_high = nypd['LATITUDE'].max()
latitude_low = nypd['LATITUDE'].min()
longitude_high = nypd['LONGITUDE'].max()
longitude_low = nypd['LONGITUDE'].min()

(latitude_low, latitude_high), (longitude_low, longitude_high)


# In[4]:


nypd['latti_range'] = pd.cut(nypd['LATITUDE'], 10)
nypd['longi_range'] = pd.cut(nypd['LONGITUDE'], 10)
nypd.head()


# In[5]:


a = nypd['latti_range'].unique()
sorted(a)


# In[32]:


# read all intersection coordinates data
intersections = pd.read_csv('intersection.csv')


# In[ ]:


# Round location data
intersections.latitude = round(intersections.latitude,7)
intersections.longitude = round(intersections.longitude,7)


# In[33]:


# Testing data:
# latitude_low = 0
# latitude_high = 80
# longitude_low = -80
# longitude_high = 0
# test_accidents = [(70,-70),(71,-71),(69,-71),(40,-60),(20,-70),(60,-10),(40,-20),(41,-21),(40,-21),(39,-19)]
# test_inters = ['70,-70','20,-50','40,-20']

## given accident coordinate n, find the bucket it belongs to
def find_index(low, high, num_bucket, n):
    bucket_size = (high - low) / num_bucket
    index = int((n - low) / bucket_size)
    return min(index, num_bucket-1)

# compute the starting coordinate of the nth bucket
def bucket_start(low, high, num_bucket, n):
    bucket_size = (high - low) / num_bucket
    return low + bucket_size * n

# assign all accidents coordinates into total of num_bucket**2 buckets 
def accidents_in_buckets(num_bucket):
    accidents = []
    for i in range(num_bucket):
        accidents.append([])
        for j in range(num_bucket):
            accidents[i].append([])

    for i in range(len(nypd)):
        lat = nypd.iloc[i, 0]
        long = nypd.iloc[i, 1]
        bucket_idx_x = find_index(latitude_low, latitude_high, num_bucket, lat)
        bucket_idx_y = find_index(-longitude_high, -longitude_low, num_bucket, -long)
        accidents[bucket_idx_x][bucket_idx_y].append((lat,long))

    return accidents

# check if the accident is in the picture
def accident_in_picture(accident, lat_low, lat_high, long_low, long_high):
    return accident[0] >= lat_low and accident[0] <= lat_high and accident[1] >= long_low and accident[1] <= long_high
    
# find the accident count around each intersection
def get_intersection_accident_count(accidents, num_bucket, intersection_cords, picture_len):
    
    result = {} # (int, int) -> int
    total_count = 0
    
    for i in range(0, len(intersections)):
        inter_lat = intersections.iloc[i, 0]
        inter_long = intersections.iloc[i, 1]
        
        # Calculate picture boundary
        picture_lat_low = inter_lat - picture_len / 2
        picture_lat_high = inter_lat + picture_len / 2
        picture_long_low = inter_long - picture_len /2
        picture_long_high = inter_long + picture_len /2
        
        # Calculate relevant bucket
        start_bucket_x = find_index(latitude_low, latitude_high, num_bucket, picture_lat_low)
        end_bucket_x = find_index(latitude_low, latitude_high, num_bucket, picture_lat_high) + 1
        start_bucket_y = find_index(-longitude_high, -longitude_low, num_bucket, -picture_long_high)
        end_bucket_y = find_index(-longitude_high, -longitude_low, num_bucket, -picture_long_low) + 1
        accident_count = 0
        for i in range(start_bucket_x, end_bucket_x):
            for j in range(start_bucket_y, end_bucket_y):
                relevant_accidents = accidents[i][j]
                for accident in relevant_accidents:
                    if accident_in_picture(accident, picture_lat_low,
                        picture_lat_high, picture_long_low, picture_long_high):
                        accident_count += 1
        result[(inter_lat, inter_long)] = accident_count
        total_count += accident_count
    
    print("Sum of accidents around all intersections: " + str(total_count))
    return result

pic_length = 0.0001
num_bucket = 50
accidents = accidents_in_buckets(num_bucket)    
dic = get_intersection_accident_count(accidents, num_bucket, intersections, pic_length)


# In[34]:


df = pd.DataFrame(list(dic.items()), columns=['location', 'acci_counts'])


# In[35]:


df.head()


# In[36]:


df.to_csv('accidents_count.csv',index=False)

