import pandas as pd
import numpy as np
from tile_system import TileSystem as tsm # Based on https://github.com/buckhx/QuadKey - with slight modification

# ----------------------------------------------

# Converting lonlat to quadkey 
def lonlat_to_quadkey(lonlat,level) :
    pix = tsm.geo_to_pixel(lonlat,level)
    pix = np.column_stack((pix[0],pix[1]))
    tile = tsm.pixel_to_tile(pix)
    tile = np.column_stack((tile[0],tile[1]))
    qk = tsm.tile_to_quadkey(tile,level)
    qk = qk.reshape(-1,1)
    qk = pd.Series(qk[:,0]).astype(str)
    
    return qk.values

# Converting quadkey to lonlat  - Sample Based
def quadkey_to_lonlat(qk):
    tilex,tiley,level = tsm.quadkey_to_tile(qk)
    tile = np.column_stack((tilex,tiley))
    pix,piy = tsm.tile_to_pixel(tile)
    pixel= np.column_stack((pix,piy))
    lat,lon = tsm.pixel_to_geo(pixel,level)
    return np.column_stack((lat,lon))

if __name__ == "__main__": 
    latitude_list = 35 + np.random.random(size=(10,1))
    longitude_list = 139 + np.random.random(size=(10,1))
    latlon = np.column_stack((latitude_list,longitude_list))
    
    # qk17 have "'" prefix
    qk17 = lonlat_to_quadkey(latlon,17)
    qk17 = (pd.Series(qk17).str[1:]).values.astype(str)
    ll17 = quadkey_to_lonlat(qk17)
    
    # qk21 have "'" prefix
    qk21 = lonlat_to_quadkey(latlon,21)
    qk21 = (pd.Series(qk21).str[1:]).values.astype(str)
    ll21 = quadkey_to_lonlat(qk21)