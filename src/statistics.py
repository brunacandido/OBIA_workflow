import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage
import rasterio
from collections import defaultdict


def load_image_for_statistics(image_path):
    """Load Sentinel-2 image and extract bands"""
    print("Loading image for statistics...")
    
    bands = {}
    with rasterio.open(image_path) as src:
        bands['blue'] = src.read(3).astype(np.float32)
        bands['green'] = src.read(2).astype(np.float32)
        bands['red'] = src.read(1).astype(np.float32)
        bands['nir'] = src.read(4).astype(np.float32)
        
    return bands


def calculate_extended_indices(bands):
    """Calculate spectral indices including EVI"""
    indices = {}
    
    # NDVI
    ndvi_num = bands['nir'] - bands['red']
    ndvi_den = bands['nir'] + bands['red']
    indices['ndvi'] = np.divide(ndvi_num, ndvi_den, 
                               out=np.zeros_like(ndvi_num), 
                               where=ndvi_den!=0)
    
    # NDWI
    ndwi_num = bands['green'] - bands['nir']
    ndwi_den = bands['green'] + bands['nir']
    indices['ndwi'] = np.divide(ndwi_num, ndwi_den,
                               out=np.zeros_like(ndwi_num),
                               where=ndwi_den!=0)
    
    # Enhanced Vegetation Index (EVI)
    evi_num = 2.5 * (bands['nir'] - bands['red'])
    evi_den = bands['nir'] + 6 * bands['red'] - 7.5 * bands['blue'] + 1
    indices['evi'] = np.divide(evi_num, evi_den,
                              out=np.zeros_like(evi_num),
                              where=evi_den!=0)
    
    return indices


def extract_objects_from_segments(segments, indices=None):
    """Extract individual objects from segments"""
    print("Extracting objects...")
    
    if segments is None:
        # Simple threshold segmentation if no segments provided
        if indices is None or 'ndvi' not in indices:
            raise ValueError("Need either segments or indices with NDVI")
        binary = indices['ndvi'] > 0.3
        binary = morphology.opening(binary, morphology.disk(3))
        segments = measure.label(binary)
        
    # Get object properties
    objects = measure.regionprops(segments)
    print(f"Found {len(objects)} objects")
    
    return objects, segments


def calculate_spectral_statistics(objects, segments, bands, indices):
    """Calculate spectral statistics for each object"""
    print("Calculating spectral statistics...")
    
    stats_list = []
    
    for obj in objects:
        if obj.area < 10:  # Skip very small objects
            continue
            
        mask = segments == obj.label
        stats = {'object_id': obj.label}
        
        # Basic spectral statistics
        for band_name, band_data in bands.items():
            values = band_data[mask]
            stats[f'{band_name}_mean'] = np.mean(values)
            stats[f'{band_name}_std'] = np.std(values)
            stats[f'{band_name}_median'] = np.median(values)
            
        # Index statistics
        for idx_name, idx_data in indices.items():
            values = idx_data[mask]
            stats[f'{idx_name}_mean'] = np.mean(values)
            stats[f'{idx_name}_std'] = np.std(values)
            
        stats_list.append(stats)
        
    return pd.DataFrame(stats_list)


def calculate_geometric_statistics(objects):
    """Calculate geometric/shape statistics"""
    print("Calculating geometric statistics...")
    
    geo_stats = []
    
    for obj in objects:
        if obj.area < 10:
            continue
            
        stats = {
            'object_id': obj.label,
            'area': obj.area,
            'perimeter': obj.perimeter,
            'compactness': (obj.perimeter ** 2) / (4 * np.pi * obj.area),
            'eccentricity': obj.eccentricity,
            'solidity': obj.solidity,
            'extent': obj.extent,
            'aspect_ratio': obj.major_axis_length / obj.minor_axis_length if obj.minor_axis_length > 0 else 0
        }
        
        geo_stats.append(stats)
        
    return pd.DataFrame(geo_stats)


def calculate_texture_statistics(objects, segments, bands):
    """Calculate basic texture statistics"""
    print("Calculating texture statistics...")
    
    texture_stats = []
    
    # Use NIR band for texture analysis
    nir_band = bands['nir']
    
    for obj in objects:
        if obj.area < 10:
            continue
            
        mask = segments == obj.label
        
        # Get bounding box for local calculations
        minr, minc, maxr, maxc = obj.bbox
        local_mask = mask[minr:maxr, minc:maxc]
        local_nir = nir_band[minr:maxr, minc:maxc]
        
        # Simple texture measures
        if np.sum(local_mask) > 0:
            # Local variance (texture measure)
            local_var = ndimage.generic_filter(local_nir, np.var, size=3)
            texture_variance = np.mean(local_var[local_mask])
            
            # Range
            texture_range = np.ptp(local_nir[local_mask])
            
            stats = {
                'object_id': obj.label,
                'texture_variance': texture_variance,
                'texture_range': texture_range,
                'homogeneity': 1 / (1 + texture_variance) if texture_variance > 0 else 1
            }
            
            texture_stats.append(stats)
            
    return pd.DataFrame(texture_stats)


def merge_all_statistics(spectral_df, geometric_df, texture_df):
    """Combine all statistics into single dataframe"""
    print("Merging all statistics...")
    
    # Merge dataframes
    stats_df = spectral_df
    stats_df = stats_df.merge(geometric_df, on='object_id', how='left')
    stats_df = stats_df.merge(texture_df, on='object_id', how='left')
    
    print(f"Statistics calculated for {len(stats_df)} objects")
    return stats_df


def classify_objects(stats_df):
    """Simple rule-based classification of objects"""
    conditions = [
        (stats_df['ndvi_mean'] > 0.6) & (stats_df['area'] > 100),
        (stats_df['ndwi_mean'] > 0.3) & (stats_df['ndvi_mean'] < 0.2),
        (stats_df['ndvi_mean'] < 0.2) & (stats_df['red_mean'] > stats_df['nir_mean']),
        (stats_df['compactness'] < 1.5) & (stats_df['area'] > 500)
    ]
    
    choices = ['Dense Vegetation', 'Water', 'Built-up/Bare', 'Large Structure']
    
    stats_df['class'] = np.select(conditions, choices, default='Mixed/Other')
    
    return stats_df['class'].value_counts()


def plot_obia_statistics(stats_df):
    """Visualize key statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('OBIA Statistics Dashboard', fontsize=16)
    
    # NDVI vs Area
    axes[0, 0].scatter(stats_df['area'], stats_df['ndvi_mean'], alpha=0.6)
    axes[0, 0].set_xlabel('Area (pixels)')
    axes[0, 0].set_ylabel('Mean NDVI')
    axes[0, 0].set_title('NDVI vs Object Size')
    
    # Compactness distribution
    axes[0, 1].hist(stats_df['compactness'], bins=20, alpha=0.7)
    axes[0, 1].set_xlabel('Compactness')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Object Compactness Distribution')
    
    # Texture variance vs NDVI
    if 'texture_variance' in stats_df.columns:
        axes[0, 2].scatter(stats_df['texture_variance'], 
                         stats_df['ndvi_mean'], alpha=0.6)
        axes[0, 2].set_xlabel('Texture Variance')
        axes[0, 2].set_ylabel('Mean NDVI')
        axes[0, 2].set_title('Texture vs Vegetation')
    
    # Classification results
    if 'class' in stats_df.columns:
        class_counts = stats_df['class'].value_counts()
        axes[1, 0].bar(range(len(class_counts)), class_counts.values)
        axes[1, 0].set_xticks(range(len(class_counts)))
        axes[1, 0].set_xticklabels(class_counts.index, rotation=45)
        axes[1, 0].set_title('Object Classification')
    
    # Spectral signature comparison
    spectral_cols = [col for col in stats_df.columns if col.endswith('_mean') and 
                    col.split('_')[0] in ['blue', 'green', 'red', 'nir']]
    
    if len(spectral_cols) > 0:
        mean_signature = stats_df[spectral_cols].mean()
        axes[1, 1].plot(range(len(mean_signature)), mean_signature.values, 'o-')
        axes[1, 1].set_xticks(range(len(mean_signature)))
        axes[1, 1].set_xticklabels([col.replace('_mean', '') for col in spectral_cols])
        axes[1, 1].set_title('Average Spectral Signature')
    
    # Size distribution
    axes[1, 2].hist(np.log10(stats_df['area']), bins=20, alpha=0.7)
    axes[1, 2].set_xlabel('Log10(Area)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Object Size Distribution')
    
    plt.tight_layout()
    plt.show()


def export_statistics(stats_df, output_path):
    """Export statistics to CSV"""
    stats_df.to_csv(output_path, index=False)
    print(f"Statistics exported to {output_path}")


def generate_summary_report(stats_df):
    """Generate summary report"""
    print("\n=== OBIA STATISTICS SUMMARY ===")
    print(f"Total objects analyzed: {len(stats_df)}")
    print(f"Total area covered: {stats_df['area'].sum()} pixels")
    print(f"Average object size: {stats_df['area'].mean():.1f} pixels")
    print(f"Size range: {stats_df['area'].min()} - {stats_df['area'].max()} pixels")
    
    print(f"\nSpectral Statistics:")
    print(f"Mean NDVI: {stats_df['ndvi_mean'].mean():.3f} ± {stats_df['ndvi_mean'].std():.3f}")
    print(f"Mean NDWI: {stats_df['ndwi_mean'].mean():.3f} ± {stats_df['ndwi_mean'].std():.3f}")
    
    if 'class' in stats_df.columns:
        print(f"\nClassification Results:")
        for class_name, count in stats_df['class'].value_counts().items():
            percentage = (count / len(stats_df)) * 100
            print(f"  {class_name}: {count} objects ({percentage:.1f}%)")


def run_statistics_workflow(image_path, segments=None):
    """
    Run complete OBIA statistics workflow
    
    Args:
        image_path: Path to Sentinel-2 image
        segments: Optional segmentation array
    """
    
    # Load and process
    bands = load_image_for_statistics(image_path)
    indices = calculate_extended_indices(bands)
    objects, segments = extract_objects_from_segments(segments, indices)
    
    # Calculate all statistics
    spectral_df = calculate_spectral_statistics(objects, segments, bands, indices)
    geometric_df = calculate_geometric_statistics(objects)
    texture_df = calculate_texture_statistics(objects, segments, bands)
    
    stats_df = merge_all_statistics(spectral_df, geometric_df, texture_df)
    
    # Classify objects
    class_counts = classify_objects(stats_df)
    
    # Generate visualizations and summary
    plot_obia_statistics(stats_df)
    generate_summary_report(stats_df)
    
    return stats_df, objects, segments

# Example usage:
# stats_df, objects, segments = run_obia_statistics('/Users/devseed/Documents/repos/OBIA_workflow/data/ortho_subset_I.tif')
# export_statistics(stats_df, 'obia_statistics.csv')