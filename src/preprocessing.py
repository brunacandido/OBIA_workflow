### preprocessing.py

import os
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd


def load_band(path):
    with rasterio.open(path) as src:
        band = src.read(1)
        profile = src.profile
    return band, profile


def normalize(array):
    array = array.astype('float32')
    return (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-10)


def calculate_ndvi(nir, red):
    nir = nir.astype(
    red = red.astype('float32')
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi


def clip_raster_with_shapefile(raster_path, shapefile_path):
    with rasterio.open(raster_path) as src:
        shapefile = gpd.read_file(shapefile_path)
        shapes = [feature["geometry"] for feature in shapefile.iterfeatures()]
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    return out_image[0], out_meta


def resample_band(band_path, scale_factor):
    with rasterio.open(band_path) as src:
        data = src.read(
            out_shape=(
                src.count,
                int(src.height * scale_factor),
                int(src.width * scale_factor)
            ),
            resampling=Resampling.bilinear
        )
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )
        profile = src.profile
        profile.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform
        })
    return data[0], profile


# Example usage:
# red, _ = load_band("../data/raw/B04.tif")
# nir, _ = load_band("../data/raw/B08.tif")
# ndvi = calculate_ndvi(nir, red)
# clipped_ndvi, ndvi_meta = clip_raster_with_shapefile("../data/raw/B08.tif", "../data/aoi.shp")
# norm_red = normalize(red)
# resampled_band, resampled_meta = resample_band("../data/raw/B02.tif", 0.5)
