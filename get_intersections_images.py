
# This file first extracts the coordinates of all the intersections in Manhattan from OpenStreetMap API
# Then it uses these intersection coordinates as a key to download satellite image from Google Map API.
# It returns lists of coordinates and PNG images of every intersection.

# coding: utf-8

# ## get center coordinate

# In[5]:


import xml.etree.ElementTree as ET


# In[6]:


def extract_intersections(osm, verbose=True):
    # This function takes an osm file as an input. It then goes through each xml 
    # element and searches for nodes that are shared by two or more ways.
    # Parameter:
    # - osm: An xml file downloaded from OpenStreetMap's map API.
    # - verbose: If true, print some outputs to terminal.
    # 
    # Ex) extract_intersections('WashingtonDC.osm')
    #
    tree = ET.parse(osm)
    root = tree.getroot()
    counter = {}
    for child in root:
        #if child.tag == 'bounds':
         #   print(child.attrib)
        if child.tag == 'way'or child.tag == 'node':
            for item in child:
                #print(item.tag)
                #print(item.attrib)
                if item.tag == 'nd':
                    nd_ref = item.attrib['ref']
                    if not nd_ref in counter:
                        counter[nd_ref] = 0
                    counter[nd_ref] += 1

    # Find nodes that are shared with more than one way, which
    # might correspond to intersections
    intersections = filter(lambda x: counter[x] > 1,  counter)
    #print(counter)

    # Extract intersection coordinates
    # You can plot the result using this url.
    # http://www.darrinward.com/lat-long/
    intersection_coordinates = []
    for child in root:
        #print(child.attrib)
        #print(child)

        if child.tag == 'node' or child.tag == 'way'and child.attrib['id'] in intersections:
            for item in child:
                if item.tag == 'tag':
                    #print(item.attrib)
                    tag_type = item.attrib['k']
                    #print(tag_type)
                    if 'highway' in tag_type:
                        #if child.attrib['id'] in intersections:
                        coordinate = child.attrib['lat'] + ',' + child.attrib['lon']
                        if verbose:
                            print(coordinate)
                        intersection_coordinates.append(coordinate)
                    else:
                        continue

    return intersection_coordinates


# In[7]:


map_list = ['[-73.9608,40.8352,-73.9339,40.8196].osm',
            '[-73.9693,40.8196,-73.9332,40.8078].osm',
            '[-73.9772,40.8078,-73.9273,40.7978].osm',
            '[-73.9845,40.7978,-73.9277,40.7878].osm',
            '[-73.9859,40.7378,-73.9703,40.7092].osm',
            '[-73.9906,40.7878,-73.9373,40.7778].osm',
            '[-73.9970,40.7312,-73.9859,40.7069].osm',
            '[-73.9986,40.7778,-73.9413,40.7678].osm',
            '[-74.0072,40.7678,-73.9489,40.7578].osm',
            '[-74.0116,40.7578,-73.9587,40.7478].osm',
            '[-74.0154,40.7478,-73.9675,40.7378].osm',
            '[-74.0200,40.7312,-73.9970,40.6997].osm'
           ]


# In[8]:


cord = []
for map in map_list:
    cord = cord + extract_intersections(osm = map, verbose = False)


# In[9]:


len(cord)


# In[10]:


cord_uni = list(set(cord))
len(cord_uni)


# In[109]:


print(cord)


# ## GET MAP

# In[113]:


from PIL import Image
import urllib.request
import numpy as np
from io import StringIO
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import cv2


# In[110]:


def get_url(size,centerLat,centerLon,zoom,key):
    url = 'http://maps.googleapis.com/maps/api/staticmap?sensor=false'
        +'&size='+str(size)+'x'+str(size)+'&center='+str(centerLat)+','
        +str(centerLon)+'&zoom='+str(zoom)
        +'&maptype=satellite'+'&key='+ key
    return url
def download_image(url,centerLat,centerLon):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    #image.show()
    image.save('Saved Image/'+str(centerLat)+'_'+str(centerLon)+'.png','PNG')


# In[ ]:


## Warning: Will take a long time!

size = 640
zoom = 20
key = 'AIzaSyBcm6xXVf2H88_FExXWjn2HN8k9YOnSMJk'
for center in center_list:
    center_lat = center.split(',')[0]
    center_lon = center.split(',')[1]
    url = get_url(size,center_lat,center_lon,zoom,key)
    download_image(url,center_lat,center_lon)

