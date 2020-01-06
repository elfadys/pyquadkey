import pandas as pd
import numpy as np
from tile_system import TileSystem as tsm # Based on https://github.com/buckhx/QuadKey - with slight modification


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
    pixel = np.column_stack((pix,piy))
    lat,lon = tsm.pixel_to_geo(pixel,level)
    return np.column_stack((lat,lon))


# Get list of quadkey based on shift value in loshift
def lonlat_to_quadkey_with_shift(quadkey, loshift):
    quadkeys = np.array([quadkey] * len(loshift))
    tilex, tiley, level = tsm.quadkey_to_tile(quadkeys)
    tile = np.column_stack((tilex, tiley))
    shift = np.array(loshift)
    tile = tile + shift
    loquadkey = tsm.tile_to_quadkey(tile, level)
    loquadkey = loquadkey.reshape(-1, 1)
    loquadkey = pd.Series(loquadkey[:, 0]).astype(str)
    loquadkey = loquadkey.str[1:]

    return loquadkey.values.astype(str)


# Converting lonlat to pixel
# See far below on how to use
def lonlat_to_metre(lonlat, level):
    pix = tsm.geo_to_pixel(lonlat, level)
    pix = np.column_stack((pix[0], pix[1]))

    grd_res = tsm.ground_resolution(lonlat[:, 0], level)
    rel_metre = pix * grd_res.reshape(-1, 1)

    return rel_metre, grd_res


# Converting quadkey to metre
# Note : use grd_res from lonlat_to_metre() function.
def quadkey_to_metre(qk, grd_res):
    tilex, tiley, level = tsm.quadkey_to_tile(qk)
    tile = np.column_stack((tilex, tiley))
    pix, piy = tsm.tile_to_pixel(tile)
    pixel = np.column_stack((pix, piy))

    rel_metre = pixel * grd_res.reshape(-1, 1)

    return rel_metre


if __name__ == "__main__":
    latitude_list = 35 + np.random.random(size=(10, 1))
    longitude_list = 139 + np.random.random(size=(10, 1))
    latlon = np.column_stack((latitude_list, longitude_list))

    # qk17 have "'" prefix
    qk17 = lonlat_to_quadkey(latlon, 17)
    qk17 = (pd.Series(qk17).str[1:]).values.astype(str)
    ll17 = quadkey_to_lonlat(qk17)
    print('Sample Level 17 : \n',np.column_stack((qk17,ll17)))

    # qk21 have "'" prefix
    qk21 = lonlat_to_quadkey(latlon, 21)
    qk21 = (pd.Series(qk21).str[1:]).values.astype(str)
    ll21 = quadkey_to_lonlat(qk21)
    print('Sample Level 21 : \n',np.column_stack((qk21,ll21)))

    # This far below.
    # qk have "'" prefix
    qk17 = lonlat_to_quadkey(latlon, 17)
    llm_17, grd_res_17 = lonlat_to_metre(latlon, 17)
    qk17 = (pd.Series(qk17).str[1:]).values.astype(str)
    qkm_17 = quadkey_to_metre(qk17, grd_res_17)
    delta_m = llm_17 - qkm_17
    print('Distance from Point to QK-start-point : \n', delta_m)

    sample_quadkey = '1200'
    sample_loshift = [[1, 0], [1, 1], [0, 1]]
    sample_loquadkey = lonlat_to_quadkey_with_shift(sample_quadkey, sample_loshift)
    print('Neighboring QK {} : \n{}'.format(sample_quadkey,sample_loquadkey))