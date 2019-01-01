# This file is an example of getting a satellite image from Google Map API

# coding: utf-8

# In[1]:


from PIL import Image
import urllib.request
import numpy as np
from io import StringIO
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import cv2


# In[2]:


# Setup parameters for the map image
centerLat,centerLon = (40.730535,-73.992436); zoom = 16; size = 640


# In[3]:


# Get url
key = 'AIzaSyBcm6xXVf2H88_FExXWjn2HN8k9YOnSMJk'
url = 'http://maps.googleapis.com/maps/api/staticmap?sensor=false'
	+'&size='+str(size)+'x'+str(size)+'&center='+str(centerLat)+','
	+str(centerLon)+'&zoom='+str(zoom)
	+'&maptype=satellite'+'&key='+ key
print(url)


# In[4]:


# Download image from the url. Save as PNG
response = requests.get(url)
image = Image.open(BytesIO(response.content))
image.show()
image.save('testimage'+'.png','PNG')


# In[ ]:


image.size


# In[ ]:


image.mode


# In[ ]:


# Need to convert to mode 'RGB' to split to 3 tunnels
image_rgb= image.convert("RGB")


# In[ ]:


# Split tunnels
r,g,b = image_rgb.split()


# In[ ]:


# Convert the image(pixels) to numbers
# Take a look at values in the original image

#image_array = np.array(image)
#print(image_array)


# In[ ]:


# Take a look at values in the splited image: Each value in the original matrix splits into a 3*1 array

#image_rgb_array = np.array(image_rgb)
#print(image_rgb_array)


# In[ ]:


# What we want to use in training 
# Also take a look at values in red tunnel (Each takes the first value from the splited 3*1 array)
r_array = np.array(r)
g_array = np.array(g)
b_array = np.array(b)
#print(r_array)


# In[ ]:


# Crop the image into 9*9 sub images
# Double for loop may be very slow in larger datasets

# index = np.linspace(0,640,num = 9)
# for i in range(0,8):
#     for j in range(0,8):
#         box = (index[i],index[j],index[i+1],index[j+1])
#         sub = image.crop(box)
#         sub.save('Saved Image/sub'+str(i)+'_'+str(j)+'.png','PNG')

# no use, since google API can only provide 640*640 image

