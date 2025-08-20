import rasterio
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.segmentation import mark_boundaries, slic, watershed
from skimage import filters
from skimage.measure import regionprops_table, perimeter
from skimage.util import map_array
import pandas as pd

def load_image(path):
    with rasterio.open(path) as src:
        bands = src.read([1, 2, 3, 4]).astype(float) / 255
        red, green, blue, nir = bands
    return bands

def calculate_rgb(red, green, blue):
    rgb = np.stack([red, green, blue], axis=-1)
    rgb_shape = rgb.shape
    rgb_size = rgb.size
    return rgb, rgb_shape, rgb_size

def calculate_ndvi(red, nir):
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi_shape = ndvi.shape
    ndvi_size = ndvi.size
    return ndvi, ndvi_shape, ndvi_size

def display_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def segmentation(image, segments, compactness, seg_mode, markers=2000):
    # seg_mode can be:
    # - slic
    # - canny
    # - sobel
    # - este
    # - este
    # - este
    if seg_mode == "slic":
        segmentation = slic(image, 
                            segments, 
                            compactness)
        boundaries = mark_boundaries(image, 
                                     segmentation, 
                                     color=(1, 0, 0), 
                                     mode="thick")
    
    if seg_mode == "canny":
        canny = skimage.feature.canny(skimage.color.rgb2gray(image))
        segmentation = watershed(canny,
                                 markers=markers)
        boundaries = mark_boundaries(image, 
                                     segmentation, 
                                     color=(1, 0, 0), 
                                     mode="thick")
        
    if seg_mode == "sobel":
        sobel = skimage.filters.sobel(skimage.color.rgb2gray(image))
        segmentation = watershed(sobel,
                                 markers=markers)
        boundaries = mark_boundaries(image, 
                                     segmentation, 
                                     color=(1, 0, 0), 
                                     mode="thick")
    return segmentation, boundaries


