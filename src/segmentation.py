import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import measure, morphology
import rasterio


def load_sentinel2_image(image_path):
    """Load Sentinel-2 image and extract relevant bands"""
    print("Loading Sentinel-2 image...")
    
    with rasterio.open(image_path) as src:       
        bands = {}
        bands['blue'] = src.read(3).astype(np.float32)
        bands['green'] = src.read(2).astype(np.float32)
        bands['red'] = src.read(1).astype(np.float32)
        bands['nir'] = src.read(4).astype(np.float32)
        
        profile = src.profile
        
    print(f"Image loaded: {bands['red'].shape}")
    return bands, profile


def calculate_spectral_indices(bands):
    """Calculate vegetation and water indices"""
    print("Calculating spectral indices...")
    
    indices = {}
    
    ndvi_num = bands['nir'] - bands['red']
    ndvi_den = bands['nir'] + bands['red']
    indices['ndvi'] = np.divide(ndvi_num, ndvi_den, 
                               out=np.zeros_like(ndvi_num), 
                               where=ndvi_den!=0)
    
    ndwi_num = bands['green'] - bands['nir']
    ndwi_den = bands['green'] + bands['nir']
    indices['ndwi'] = np.divide(ndwi_num, ndwi_den,
                               out=np.zeros_like(ndwi_num),
                               where=ndwi_den!=0)
    
    indices['brightness'] = (bands['red'] + 
                           bands['green'] + 
                           bands['blue']) / 3
    
    return indices


def threshold_segmentation(indices):
    """Simple threshold-based segmentation"""
    print("Performing threshold segmentation...")
    
    segments = np.zeros_like(indices['ndvi'], dtype=np.uint8)
    
    water_mask = (indices['ndwi'] > 0.3) & (indices['ndvi'] < 0.2)
    segments[water_mask] = 1
    
    vegetation_mask = indices['ndvi'] > 0.4
    segments[vegetation_mask] = 2
    
    built_up_mask = (indices['ndvi'] < 0.2) & (indices['brightness'] > 1000)
    segments[built_up_mask] = 3
    
    segments = morphology.opening(segments, morphology.disk(2))
    segments = morphology.closing(segments, morphology.disk(3))
    
    return segments


def kmeans_segmentation(bands, indices, n_clusters=5):
    """K-means clustering based segmentation"""
    print(f"Performing K-means segmentation with {n_clusters} clusters...")
    
    features = np.stack([
        bands['red'].flatten(),
        bands['green'].flatten(), 
        bands['blue'].flatten(),
        bands['nir'].flatten(),
        indices['ndvi'].flatten(),
        indices['ndwi'].flatten()
    ], axis=1)
    
    valid_mask = np.all(np.isfinite(features), axis=1)
    features_clean = features[valid_mask]
    
    features_norm = (features_clean - features_clean.mean(axis=0)) / features_clean.std(axis=0)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_norm)
    
    segments = np.zeros(bands['red'].size, dtype=np.uint8)
    segments[valid_mask] = labels + 1
    segments = segments.reshape(bands['red'].shape)
    
    return segments


def edge_based_segmentation(indices):
    """Edge-based segmentation using watershed"""
    print("Performing edge-based segmentation...")
    
    ndvi_smooth = filters.gaussian(indices['ndvi'], sigma=1)
    edges = filters.sobel(ndvi_smooth)
    
    markers = np.zeros_like(indices['ndvi'], dtype=np.int32)
    
    markers[indices['ndvi'] > 0.6] = 1
    markers[indices['ndwi'] > 0.4] = 2
    markers[(indices['ndvi'] < 0.2) & (indices['brightness'] > 1200)] = 3
    
    segments = segmentation.watershed(edges, markers)
    
    return segments.astype(np.uint8)


def visualize_segmentation_results(bands, indices, segments, method_name="Segmentation"):
    """Visualize the segmentation results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Sentinel-2 {method_name} Results', fontsize=16)
    
    rgb = np.stack([bands['red'], bands['green'], bands['blue']], axis=2)
    rgb_norm = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
    axes[0, 0].imshow(rgb_norm)
    axes[0, 0].set_title('RGB Composite')
    axes[0, 0].axis('off')
    
    ndvi_plot = axes[0, 1].imshow(indices['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0, 1].set_title('NDVI')
    axes[0, 1].axis('off')
    plt.colorbar(ndvi_plot, ax=axes[0, 1], fraction=0.046)
    
    ndwi_plot = axes[0, 2].imshow(indices['ndwi'], cmap='Blues', vmin=-1, vmax=1)
    axes[0, 2].set_title('NDWI') 
    axes[0, 2].axis('off')
    plt.colorbar(ndwi_plot, ax=axes[0, 2], fraction=0.046)
    
    seg_plot = axes[1, 0].imshow(segments, cmap='tab10')
    axes[1, 0].set_title('Segmentation')
    axes[1, 0].axis('off')
    plt.colorbar(seg_plot, ax=axes[1, 0], fraction=0.046)
    
    axes[1, 1].imshow(rgb_norm)
    axes[1, 1].imshow(segments, alpha=0.5, cmap='tab10')
    axes[1, 1].set_title('Overlay on RGB')
    axes[1, 1].axis('off')
    
    unique, counts = np.unique(segments[segments > 0], return_counts=True)
    axes[1, 2].bar(unique, counts)
    axes[1, 2].set_title('Segment Statistics')
    axes[1, 2].set_xlabel('Segment ID')
    axes[1, 2].set_ylabel('Pixel Count')
    
    plt.tight_layout()
    plt.show()


def get_segment_statistics(segments, indices):
    """Get statistics for each segment"""
    stats = {}
    unique_segments = np.unique(segments[segments > 0])
    
    for seg_id in unique_segments:
        mask = segments == seg_id
        stats[f'segment_{seg_id}'] = {
            'pixel_count': np.sum(mask),
            'mean_ndvi': np.mean(indices['ndvi'][mask]),
            'mean_ndwi': np.mean(indices['ndwi'][mask]),
            'mean_brightness': np.mean(indices['brightness'][mask])
        }
        
    return stats


def run_segmentation_workflow(image_path, method='threshold'):
    """
    Run the complete segmentation workflow
    
    Args:
        image_path: Path to Sentinel-2 image
        method: 'threshold', 'kmeans', or 'edge'
    """
    
    bands, profile = load_sentinel2_image(image_path)
    indices = calculate_spectral_indices(bands)
    
    if method == 'threshold':
        segments = threshold_segmentation(indices)
    elif method == 'kmeans':
        segments = kmeans_segmentation(bands, indices, n_clusters=6)
    elif method == 'edge':
        segments = edge_based_segmentation(indices)
    
    visualize_segmentation_results(bands, indices, segments, 
                                  method_name=f"{method.capitalize()} Segmentation")
    
    stats = get_segment_statistics(segments, indices)
    print("\nSegmentation Statistics:")
    for segment, data in stats.items():
        print(f"{segment}: {data['pixel_count']} pixels, "
              f"NDVI: {data['mean_ndvi']:.3f}, "
              f"NDWI: {data['mean_ndwi']:.3f}")
    
    return bands, indices, segments, profile

# Example usage:
# bands, indices, segments, profile = run_segmentation_workflow('/Users/devseed/Documents/repos/OBIA_workflow/data/sample.tif', method='edge')
# export_segmentation_results(segments, profile, 'segmentation_results.tif')

def manual_treshold_segmentation(ndvi, 
                                 threshold1, 
                                 threshold2, 
                                 threshold3,
                                 treshold4):
    thresholds = [threshold1, threshold2, threshold3, treshold4]
    for i, threshold in enumerate(thresholds):
        mask = ndvi > threshold
    return mask