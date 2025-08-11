import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from skimage import morphology, filters
import rasterio
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_image_for_classification(image_path):
    """Load Sentinel-2 image and extract relevant bands"""
    print("Loading Sentinel-2 image for classification...")
    
    with rasterio.open(image_path) as src:       
        bands = {}
        bands['blue'] = src.read(3).astype(np.float32)
        bands['green'] = src.read(2).astype(np.float32)
        bands['red'] = src.read(1).astype(np.float32)
        bands['nir'] = src.read(4).astype(np.float32)
        
        profile = src.profile
        
    print(f"Image loaded: {bands['red'].shape}")
    return bands, profile


def calculate_classification_indices(bands):
    """Calculate spectral indices for classification"""
    print("Calculating spectral indices for classification...")
    
    indices = {}
    
    # NDVI - Vegetation index
    ndvi_num = bands['nir'] - bands['red']
    ndvi_den = bands['nir'] + bands['red']
    indices['ndvi'] = np.divide(ndvi_num, ndvi_den, 
                               out=np.zeros_like(ndvi_num), 
                               where=ndvi_den!=0)
    
    # NDWI - Water index
    ndwi_num = bands['green'] - bands['nir']
    ndwi_den = bands['green'] + bands['nir']
    indices['ndwi'] = np.divide(ndwi_num, ndwi_den,
                               out=np.zeros_like(ndwi_num),
                               where=ndwi_den!=0)
    
    # NDBI - Built-up index
    ndbi_num = bands['red'] - bands['nir']  # Using red instead of SWIR
    ndbi_den = bands['red'] + bands['nir']
    indices['ndbi'] = np.divide(ndbi_num, ndbi_den,
                               out=np.zeros_like(ndbi_num),
                               where=ndbi_den!=0)
    
    # Brightness
    indices['brightness'] = (bands['red'] + bands['green'] + bands['blue']) / 3
    
    # NIR/Red ratio
    indices['nir_red_ratio'] = np.divide(bands['nir'], bands['red'],
                                        out=np.ones_like(bands['nir']),
                                        where=bands['red']!=0)
    
    return indices


def rule_based_classification(bands, indices):
    """
    Perform rule-based classification into basic land cover classes
    Classes: 0=Unclassified, 1=Water, 2=Vegetation, 3=Built-up/Bare, 4=Dense Vegetation
    """
    print("Performing rule-based classification...")
    
    height, width = bands['red'].shape
    classification = np.zeros((height, width), dtype=np.uint8)
    
    # Water classification (highest priority)
    water_mask = (
        (indices['ndwi'] > 0.3) & 
        (indices['ndvi'] < 0.2) & 
        (indices['brightness'] < 2000)
    )
    classification[water_mask] = 1
    
    # Dense vegetation classification
    dense_veg_mask = (
        (indices['ndvi'] > 0.6) & 
        (indices['nir_red_ratio'] > 2.5) &
        (~water_mask)  # Exclude already classified water
    )
    classification[dense_veg_mask] = 4
    
    # General vegetation classification
    vegetation_mask = (
        (indices['ndvi'] > 0.3) & 
        (indices['ndvi'] <= 0.6) &
        (indices['nir_red_ratio'] > 1.5) &
        (~water_mask) & (~dense_veg_mask)
    )
    classification[vegetation_mask] = 2
    
    # Built-up/Bare soil classification
    built_up_mask = (
        (indices['ndvi'] < 0.3) & 
        (indices['ndwi'] < 0.1) & 
        (indices['brightness'] > 1000) &
        (~water_mask)
    )
    classification[built_up_mask] = 3
    
    # Apply morphological operations to clean up classification
    classification = morphology.opening(classification, morphology.disk(2))
    classification = morphology.closing(classification, morphology.disk(3))
    
    return classification


def segment_based_classification(segments, bands, indices):
    """
    Classify segments based on their spectral properties
    """
    print("Performing segment-based classification...")
    
    classification = np.zeros_like(segments, dtype=np.uint8)
    unique_segments = np.unique(segments[segments > 0])
    
    for segment_id in unique_segments:
        mask = segments == segment_id
        
        if np.sum(mask) < 5:  # Skip very small segments
            continue
            
        # Calculate mean values for the segment
        mean_ndvi = np.mean(indices['ndvi'][mask])
        mean_ndwi = np.mean(indices['ndwi'][mask])
        mean_brightness = np.mean(indices['brightness'][mask])
        mean_nir_red = np.mean(indices['nir_red_ratio'][mask])
        
        # Classify based on spectral characteristics
        if mean_ndwi > 0.3 and mean_ndvi < 0.2:
            classification[mask] = 1  # Water
        elif mean_ndvi > 0.6 and mean_nir_red > 2.5:
            classification[mask] = 4  # Dense vegetation
        elif mean_ndvi > 0.3 and mean_nir_red > 1.5:
            classification[mask] = 2  # Vegetation
        elif mean_ndvi < 0.3 and mean_brightness > 1000:
            classification[mask] = 3  # Built-up/Bare
        # else remains unclassified (0)
    
    return classification


def post_process_classification(classification, min_area=10):
    """Post-process classification to remove small isolated pixels"""
    print("Post-processing classification...")
    
    processed = classification.copy()
    
    # Remove small areas for each class
    for class_id in [1, 2, 3, 4]:
        mask = classification == class_id
        cleaned = morphology.remove_small_objects(mask, min_size=min_area)
        processed[mask & ~cleaned] = 0  # Set small objects to unclassified
    
    return processed


def calculate_classification_statistics(classification):
    """Calculate statistics for classification results"""
    print("Calculating classification statistics...")
    
    class_names = {
        0: 'Unclassified',
        1: 'Water',
        2: 'Vegetation', 
        3: 'Built-up/Bare',
        4: 'Dense Vegetation'
    }
    
    unique, counts = np.unique(classification, return_counts=True)
    total_pixels = classification.size
    
    stats = {}
    for class_id, count in zip(unique, counts):
        stats[class_names[class_id]] = {
            'pixels': int(count),
            'percentage': (count / total_pixels) * 100,
            'area_km2': (count * 100) / 1000000  # Assuming 10m pixel size
        }
    
    return stats


def visualize_classification_results(bands, indices, classification, method_name="Classification"):
    """Visualize classification results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Sentinel-2 {method_name} Results', fontsize=16)
    
    # RGB composite
    rgb = np.stack([bands['red'], bands['green'], bands['blue']], axis=2)
    rgb_norm = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
    axes[0, 0].imshow(rgb_norm)
    axes[0, 0].set_title('RGB Composite')
    axes[0, 0].axis('off')
    
    # NDVI
    ndvi_plot = axes[0, 1].imshow(indices['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0, 1].set_title('NDVI')
    axes[0, 1].axis('off')
    plt.colorbar(ndvi_plot, ax=axes[0, 1], fraction=0.046)
    
    # NDWI
    ndwi_plot = axes[0, 2].imshow(indices['ndwi'], cmap='Blues', vmin=-1, vmax=1)
    axes[0, 2].set_title('NDWI')
    axes[0, 2].axis('off')
    plt.colorbar(ndwi_plot, ax=axes[0, 2], fraction=0.046)
    
    # Classification result
    class_colors = ['black', 'blue', 'green', 'red', 'darkgreen']
    class_cmap = plt.matplotlib.colors.ListedColormap(class_colors)
    
    class_plot = axes[1, 0].imshow(classification, cmap=class_cmap, vmin=0, vmax=4)
    axes[1, 0].set_title('Land Cover Classification')
    axes[1, 0].axis('off')
    
    # Create custom colorbar
    cbar = plt.colorbar(class_plot, ax=axes[1, 0], fraction=0.046)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Unclassified', 'Water', 'Vegetation', 'Built-up/Bare', 'Dense Veg'])
    
    # Overlay on RGB
    axes[1, 1].imshow(rgb_norm)
    masked_class = np.ma.masked_where(classification == 0, classification)
    axes[1, 1].imshow(masked_class, alpha=0.6, cmap=class_cmap, vmin=0, vmax=4)
    axes[1, 1].set_title('Classification Overlay')
    axes[1, 1].axis('off')
    
    # Class distribution
    unique, counts = np.unique(classification, return_counts=True)
    class_names = ['Unclassified', 'Water', 'Vegetation', 'Built-up/Bare', 'Dense Veg']
    colors = ['black', 'blue', 'green', 'red', 'darkgreen']
    
    bars = axes[1, 2].bar(range(len(unique)), counts, 
                         color=[colors[i] for i in unique])
    axes[1, 2].set_xticks(range(len(unique)))
    axes[1, 2].set_xticklabels([class_names[i] for i in unique], rotation=45)
    axes[1, 2].set_title('Class Distribution')
    axes[1, 2].set_ylabel('Pixel Count')
    
    plt.tight_layout()
    plt.show()


def export_classification(classification, profile, output_path):
    """Export classification to GeoTIFF"""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=classification.shape[0],
        width=classification.shape[1],
        count=1,
        dtype=classification.dtype,
        crs=profile['crs'],
        transform=profile['transform'],
    ) as dst:
        dst.write(classification, 1)
    
    print(f"Classification exported to {output_path}")


def create_classification_report(stats):
    """Create a formatted classification report"""
    print("\n" + "="*50)
    print("LAND COVER CLASSIFICATION REPORT")
    print("="*50)
    
    total_area = sum([data['area_km2'] for data in stats.values()])
    
    for class_name, data in stats.items():
        print(f"\n{class_name}:")
        print(f"  Pixels: {data['pixels']:,}")
        print(f"  Percentage: {data['percentage']:.2f}%")
        print(f"  Area: {data['area_km2']:.2f} km²")
    
    print(f"\nTotal mapped area: {total_area:.2f} km²")
    print("="*50)


def validate_classification_quality(classification, indices):
    """Basic quality assessment of classification"""
    print("Assessing classification quality...")
    
    quality_metrics = {}
    
    # Check for mixed pixels (potential misclassification)
    water_mask = classification == 1
    if np.any(water_mask):
        water_ndvi = indices['ndvi'][water_mask]
        water_quality = np.sum(water_ndvi < 0.2) / len(water_ndvi) * 100
        quality_metrics['Water classification accuracy'] = f"{water_quality:.1f}%"
    
    # Check vegetation consistency
    veg_mask = (classification == 2) | (classification == 4)
    if np.any(veg_mask):
        veg_ndvi = indices['ndvi'][veg_mask]
        veg_quality = np.sum(veg_ndvi > 0.2) / len(veg_ndvi) * 100
        quality_metrics['Vegetation classification accuracy'] = f"{veg_quality:.1f}%"
    
    # Overall coverage
    classified_pixels = np.sum(classification > 0)
    total_pixels = classification.size
    coverage = (classified_pixels / total_pixels) * 100
    quality_metrics['Classification coverage'] = f"{coverage:.1f}%"
    
    print("\nQuality Assessment:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value}")
    
    return quality_metrics


def run_classification_workflow(image_path, segments=None, method='rule_based', 
                               export_path=None):
    """
    Run the complete classification workflow
    
    Args:
        image_path: Path to Sentinel-2 image
        segments: Optional segmentation array for segment-based classification
        method: 'rule_based' or 'segment_based'
        export_path: Optional path to export classification
    """
    
    # Load data
    bands, profile = load_image_for_classification(image_path)
    indices = calculate_classification_indices(bands)
    
    # Perform classification
    if method == 'rule_based':
        classification = rule_based_classification(bands, indices)
        method_name = "Rule-based Classification"
    elif method == 'segment_based':
        if segments is None:
            raise ValueError("Segments required for segment-based classification")
        classification = segment_based_classification(segments, bands, indices)
        method_name = "Segment-based Classification"
    else:
        raise ValueError("Method must be 'rule_based' or 'segment_based'")
    
    # Post-process
    classification = post_process_classification(classification)
    
    # Calculate statistics
    stats = calculate_classification_statistics(classification)
    
    # Visualize results
    visualize_classification_results(bands, indices, classification, method_name)
    
    # Generate reports
    create_classification_report(stats)
    quality_metrics = validate_classification_quality(classification, indices)
    
    # Export if requested
    if export_path:
        export_classification(classification, profile, export_path)
    
    return classification, stats, quality_metrics


# Example usage functions
def classify_with_segments(image_path, segments):
    """Convenience function for segment-based classification"""
    return run_classification_workflow(
        image_path, 
        segments=segments, 
        method='segment_based'
    )


def classify_pixel_based(image_path):
    """Convenience function for rule-based pixel classification"""
    return run_classification_workflow(
        image_path, 
        method='rule_based'
    )


# Combined workflow with segmentation
def combined_segmentation_classification_workflow(image_path, segmentation_method='threshold'):
    """
    Run segmentation followed by classification
    Note: This requires the segmentation functions to be imported
    """
    try:
        from segmentation_workflow import run_segmentation_workflow
        
        # Run segmentation first
        print("Running segmentation workflow...")
        bands, indices, segments, profile = run_segmentation_workflow(
            image_path, method=segmentation_method
        )
        
        # Then run classification
        print("\nRunning classification workflow...")
        classification, stats, quality = run_classification_workflow(
            image_path, 
            segments=segments, 
            method='segment_based'
        )
        
        return classification, segments, stats, quality
        
    except ImportError:
        print("Segmentation workflow not available. Running pixel-based classification only.")
        return run_classification_workflow(image_path, method='rule_based')


# Example usage:
# classification, stats, quality = classify_pixel_based('/path/to/image.tif')
# classification, stats, quality = classify_with_segments('/path/to/image.tif', segments)
# results = combined_segmentation_classification_workflow('/path/to/image.tif')