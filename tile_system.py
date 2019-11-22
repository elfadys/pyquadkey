from util import precondition
from math import sin, cos, atan, exp, log, pi
import numpy as np
import pandas as pd

def valid_level(level):
    LEVEL_RANGE = (1, 23)
    return LEVEL_RANGE[0] <= level <= LEVEL_RANGE[1]


@precondition(lambda key: valid_level(len(key)))
def valid_key(key):
    return TileSystem.KEY_PATTERN.match(key) is not None


class TileSystem:

    """
    Class with static method to build quadkeys from lat, lon, levels
    see http://msdn.microsoft.com/en-us/library/bb259689.aspx
    """
    import re
    KEY_PATTERN = re.compile("^[0-3]+$")

    EARTH_RADIUS = 6378137
    LATITUDE_RANGE = (-85.05112878, 85.05112878)
    LONGITUDE_RANGE = (-180., 180.)

    @staticmethod
    @precondition(lambda n, minMax: minMax[0] <= minMax[1])
    def clip(n, minMax):
        """	Clips number to specified values """
        return np.minimum(np.maximum(n, minMax[0]), minMax[1])

    @staticmethod
    @precondition(valid_level)
    def map_size(level):
        """Determines map height and width in pixel space at level"""
        return 256 << level

    @staticmethod
    @precondition(lambda lat, lvl: valid_level(lvl))
    def ground_resolution(lat, level):
        """Gets ground res in meters / pixel"""
        lat = TileSystem.clip(lat, TileSystem.LATITUDE_RANGE)
        return cos(lat * pi / 180) * 2 * pi * TileSystem.EARTH_RADIUS / TileSystem.map_size(level)

    @staticmethod
    @precondition(lambda lat, lvl, dpi: valid_level(lvl))
    def map_scale(lat, level, dpi):
        """Gets the scale of the map expressed as ratio 1	: N. Returns N"""
        return TileSystem.ground_resolution(lat, level) * dpi / 0.0254

    @staticmethod
    @precondition(lambda geo, lvl: valid_level(lvl))
    def geo_to_pixel(geo, level):
        lat, lon = geo[:,0].astype(np.float64), geo[:,1].astype(np.float64)
        lat = TileSystem.clip(lat, TileSystem.LATITUDE_RANGE)
        lon = TileSystem.clip(lon, TileSystem.LONGITUDE_RANGE)
        x = (lon + 180) / 360
        sin_lat = np.sin(lat * pi / 180)
        y = 0.5 - np.log((1 + sin_lat) / (1 - sin_lat)) / (4 * pi)
        map_size = TileSystem.map_size(level)
        pixel_x = TileSystem.clip(x * map_size + 0.5, (0, map_size - 1))
        pixel_y = TileSystem.clip(y * map_size + 0.5, (0, map_size - 1))
        return pixel_x.astype(np.int64), pixel_y.astype(np.int64)

    @staticmethod
    @precondition(lambda pix, lvl: valid_level(lvl))
    def pixel_to_geo(pixel, level):
        """Transform from pixel to geo coordinates"""
        pixel_x = pixel[:,0]
        pixel_y = pixel[:,1]
        map_size = float(TileSystem.map_size(level))
        x = (TileSystem.clip(pixel_x, (0, map_size - 1)) / map_size) - 0.5
        y = 0.5 - (TileSystem.clip(pixel_y, (0, map_size - 1)) / map_size)
        lat = 90 - 360 * np.arctan(np.exp(-y * 2 * pi)) / pi
        lon = 360 * x
        return np.round(lat, 6), np.round(lon, 6)
        
    @staticmethod
    def pixel_to_tile(pixel):
        """Transform pixel to tile coordinates"""
        return (pixel[:,0] / 256).astype(np.int64), (pixel[:,1] / 256).astype(np.int64)

    @staticmethod
    def tile_to_pixel(tile, centered=False):
        """Transform tile to pixel coordinates"""
        pixel_x = tile[:,0] * 256
        pixel_y = tile[:,1] * 256
        if centered:
            # should clip on max map size
            pixel_x += 128
            pixel_y += 128
        return pixel_x, pixel_y

    @staticmethod
    @precondition(lambda tile, lvl: valid_level(lvl))
    def tile_to_quadkey(tile, level):
        """Transform tile coordinates to a quadkey"""
        tile_x = tile[:,0]
        tile_y = tile[:,1]
        # quadkey = np.zeros((1,len(tile_x)),dtype=np.uint64)
        quadkey = pd.DataFrame("'",index=np.arange(len(tile)),columns=['qk'])
        for i in range(level):
            bit = level - i
            digit = np.zeros_like(tile_x)
            mask = 1 << (bit - 1)  # if (bit - 1) > 0 else 1 >> (bit - 1)
            digit += ((tile_x & mask) != 0)*1
            digit += ((tile_y & mask) != 0)*2
            quadkey.qk = quadkey.qk + digit.astype(str)

        return quadkey.values.astype(str)

    @staticmethod
    def quadkey_to_tile(quadkey):
        """Transform quadkey to tile coordinates"""
        tile_x  = np.zeros((len(quadkey),1)).astype(int)
        tile_y  = np.zeros((len(quadkey),1)).astype(int)
        level = len(quadkey[0])
        qk = pd.DataFrame(quadkey,columns=['qk'])
        
        for i in range(level):
            bit = level - i
            mask = 1 << (bit - 1)
        
            tile_x[qk['qk'].str[level - bit] == '1'] |= mask
            tile_y[qk['qk'].str[level - bit] == '2'] |= mask
            tile_x[qk['qk'].str[level - bit] == '3'] |= mask
            tile_y[qk['qk'].str[level - bit] == '3'] |= mask
        
        return tile_x, tile_y, level
