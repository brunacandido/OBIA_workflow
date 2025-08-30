import rasterio
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.segmentation import mark_boundaries, slic, watershed, felzenszwalb
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
    return rgb

def calculate_ndvi(red, nir):
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi_shape = ndvi.shape
    ndvi_size = ndvi.size
    return ndvi

def display_image(image, title='', colors=[], class_names=[]):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image)
    colors = colors
    class_names = class_names
    plt.axis('off')
    plt.show()

def segmentation(image, segments, compactness, seg_mode, 
                 markers=2000,
                 scale=3.0,
                 sigma=0.95,
                 min_size=50):
    # seg_mode can be:
    # - slic
    # - canny
    # - sobel
    # - thrsehold_triangle
    # - watershed
    # - felz
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
        
    if seg_mode == "thrsehold_triangle":
        opened = skimage.morphology.opening(skimage.color.rgb2gray(image), skimage.morphology.disk(1))
        closed = skimage.morphology.closing(opened, skimage.morphology.disk(1))
        thrsehold_triangle = closed < filters.threshold_triangle(closed)
        segmentation = watershed(thrsehold_triangle,
                                 markers=markers,
                                 compactness=compactness)
        boundaries = mark_boundaries(image, 
                                     segmentation, 
                                     color=(1, 0, 0), 
                                     mode="thick")
        
    if seg_mode == "watershed":
        segmentation = watershed(skimage.color.rgb2gray(image),
                                 markers=markers)
        boundaries = mark_boundaries(image, 
                                     segmentation, 
                                     color=(1, 0, 0), 
                                     mode="thick")

    if seg_mode == "felz":
        segmentation = felzenszwalb(image,
                                    scale=scale,
                                    sigma=sigma,
                                    min_size=min_size)
        boundaries = mark_boundaries(image, 
                                     segmentation, 
                                     color=(1, 0, 0), 
                                     mode="thick")
    return segmentation, boundaries

def calculate_segment_features(red, green, blue, nir, ndvi, segmentation):
    ndvi = ndvi
    unique_labels = np.unique(segmentation)
    features_list = []
    
    for label in unique_labels:
        if label == 0: 
            continue

        mask = segmentation == label
        
        if np.sum(mask) == 0:
            continue
        
        red_vals = red[mask]
        green_vals = green[mask]
        blue_vals = blue[mask]
        nir_vals = nir[mask]
        ndvi_vals = ndvi[mask]
        
        features = {
            'segment_id': label,
            'red_mean': np.mean(red_vals),
            'green_mean': np.mean(green_vals),
            'blue_mean': np.mean(blue_vals),
            'nir_mean': np.mean(nir_vals),
            'red_std': np.std(red_vals),
            'green_std': np.std(green_vals),
            'blue_std': np.std(blue_vals),
            'nir_std': np.std(nir_vals),
            'intensity_mean': np.mean([red_vals, green_vals, blue_vals, nir_vals]),
            'ndvi_mean': np.mean(ndvi_vals),
            'ndvi_std': np.std(ndvi_vals),
            'brightness': np.mean(red_vals + green_vals + blue_vals),
            'vegetation_index': np.mean(nir_vals) / (np.mean(red_vals) + 1e-10)
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)