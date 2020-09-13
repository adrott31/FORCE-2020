# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:46:35 2020

@author: aasmu
"""

import numpy as np
import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils import shuffle
# from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# import pygmt
import netCDF4
import matplotlib
from mpl_toolkits.basemap import Basemap
from pyproj import Proj

# fig = pygmt.Figure()
# # Make a global Mollweide map with automatic ticks
# fig.basemap(region="g", projection="W8i", frame=True)
# # Plot the land as light gray
# fig.coast(land="#666666", water="skyblue")
# fig.show()

# load shapefile
sh_file = "L:\\7 Common resources\\Datasets\\NPD data\\fldAreav_geo_fldarea.shp"

# load training wells
input_file = 'D:\OneDrive - Rock Physics Technology AS\GEO-FoU\PYTHON\FORCE-2020/train.csv'
data = pd.read_csv(input_file, sep=';')
data.sample(10)

input_file = 'D:\OneDrive - Rock Physics Technology AS\GEO-FoU\PYTHON\FORCE-2020/test.csv'
data2 = pd.read_csv(input_file, sep=';')
data2.sample(10)


# location map
x_loc = data['X_LOC']
y_loc = data['Y_LOC']
x_loc2 = data2['X_LOC']
y_loc2 = data2['Y_LOC']
# plt.plot(x_loc,y_loc,'.')

# convert from utm to lat/lon
my_x = x_loc.values[~np.isnan(x_loc.values)]    # remove nan from series
my_y = y_loc.values[~np.isnan(y_loc.values)]
my_x2 = x_loc2.values[~np.isnan(x_loc2.values)]    # remove nan from series
my_y2 = y_loc2.values[~np.isnan(y_loc2.values)]

ZoneNo = "31"
myProj = Proj("+proj=utm +zone="\
              + ZoneNo+", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lon, lat = myProj(my_x, my_y, inverse=True)
lon2, lat2 = myProj(my_x2, my_y2, inverse=True)
# plt.plot(lon,lat,'.')

light_grey = '#cccccc'
dark_grey  = '#aaaaaa'


# map plotting
wid = 8E5       # 1800000   # in meters
hei = 1E6       # 2300000   # in meters
lat0 = 60; lat_ts = lat0
lon0 = 5
proj = 'tmerc'
res = 'i'   # c (crude), l (low), i (intermediate), h (high), f (full) or None
cscale = 5  # etopo downsampling

plt.figure(figsize=(12, 8))
m = Basemap(width=wid,height=hei,
            resolution=res,projection=proj,
            lat_ts=lat_ts,lat_0=lat0,lon_0=lon0)
m.etopo(scale=cscale, alpha=1)
m.fillcontinents(color='lightgray',lake_color='skyblue')
# m.drawcountries(linewidth=0.2)
# m.drawmapboundary(fill_color=None)
m.drawrivers(color='k')
m.drawcoastlines()
# m.readshapefile(shape_file)
m.drawparallels(np.arange(50.,80.,2.))
m.drawmeridians(np.arange(-10.,30.,5.))
x, y = m(lon, lat)
plt.plot(x,y,'b.')
x2, y2 = m(lon2, lat2)
plt.plot(x2,y2,'r.')