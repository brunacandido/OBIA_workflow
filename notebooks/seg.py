import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import filters, measure, morphology, segmentation
import rasterio
from rasterio.plot import show
import warnings
warnings.filterwarnings('ignore')

class Sentinel2Segmentation:
    def __init__(self, image_path):
        """
        Initialize the segmentation workflow
        
        Args:
            image_path: Path to Sentinel-2 image file
        """
        self.image_path = image_path
        self.image = None
        self.bands = {}
        self.indices = {}
        self.segments = None
        
    def load_image(self):
        """Load Sentinel-2 image and extract relevant bands"""
        print("Loading Sentinel-2 image...")
        
        with rasterio.open(self.image_path) as src:
            # Read bands (assuming standard Sentinel-2 band order)
            # Band 2: Blue (490nm)
            # Band 3: Green (560nm) 
            # Band 4: Red (665nm)
            # Band 8: NIR (842nm)
            
            self.bands['blue'] = src.read(3).astype(np.float32)
            self.bands['green'] = src.read(2).astype(np.float32)
            self.bands['red'] = src.read(1).astype(np.float32)
            self.bands['nir'] = src.read(4).astype(np.float32)
            
            self.profile = src.profile
            
        print(f"Image loaded: {self.bands['red'].shape}")
        
    def calculate_indices(self):
        """Calculate vegetation and water indices"""
        print("Calculating spectral indices...")
        
        # NDVI (Normalized Difference Vegetation Index)
        ndvi_num = self.bands['nir'] - self.bands['red']
        ndvi_den = self.bands['nir'] + self.bands['red']
        self.indices['ndvi'] = np.divide(ndvi_num, ndvi_den, 
                                       out=np.zeros_like(ndvi_num), 
                                       where=ndvi_den!=0)
        
        # NDWI (Normalized Difference Water Index)
        ndwi_num = self.bands['green'] - self.bands['nir']
        ndwi_den = self.bands['green'] + self.bands['nir']
        self.indices['ndwi'] = np.divide(ndwi_num, ndwi_den,
                                       out=np.zeros_like(ndwi_num),
                                       where=ndwi_den!=0)
        
        # Simple brightness index
        self.indices['brightness'] = (self.bands['red'] + 
                                    self.bands['green'] + 
                                    self.bands['blue']) / 3
        
    def threshold_segmentation(self):
        """Simple threshold-based segmentation"""
        print("Performing threshold segmentation...")
        
        segments = np.zeros_like(self.indices['ndvi'], dtype=np.uint8)
        
        # Water bodies (high NDWI, low NDVI)
        water_mask = (self.indices['ndwi'] > 0.3) & (self.indices['ndvi'] < 0.2)
        segments[water_mask] = 1
        
        # Vegetation (high NDVI)
        vegetation_mask = self.indices['ndvi'] > 0.4
        segments[vegetation_mask] = 2
        
        # Built-up/bare soil (low NDVI, high brightness)
        built_up_mask = (self.indices['ndvi'] < 0.2) & (self.indices['brightness'] > 1000)
        segments[built_up_mask] = 3
        
        # Apply morphological operations to clean up
        segments = morphology.opening(segments, morphology.disk(2))
        segments = morphology.closing(segments, morphology.disk(3))
        
        self.segments = segments
        
    def kmeans_segmentation(self, n_clusters=5):
        """K-means clustering based segmentation"""
        print(f"Performing K-means segmentation with {n_clusters} clusters...")
        
        # Stack all features
        features = np.stack([
            self.bands['red'].flatten(),
            self.bands['green'].flatten(), 
            self.bands['blue'].flatten(),
            self.bands['nir'].flatten(),
            self.indices['ndvi'].flatten(),
            self.indices['ndwi'].flatten()
        ], axis=1)
        
        # Remove invalid pixels
        valid_mask = np.all(np.isfinite(features), axis=1)
        features_clean = features[valid_mask]
        
        # Normalize features
        features_norm = (features_clean - features_clean.mean(axis=0)) / features_clean.std(axis=0)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_norm)
        
        # Reshape back to image dimensions
        segments = np.zeros(self.bands['red'].size, dtype=np.uint8)
        segments[valid_mask] = labels + 1
        segments = segments.reshape(self.bands['red'].shape)
        
        self.segments = segments
        
    def edge_based_segmentation(self):
        """Edge-based segmentation using watershed"""
        print("Performing edge-based segmentation...")
        
        # Use NDVI for edge detection
        ndvi_smooth = filters.gaussian(self.indices['ndvi'], sigma=1)
        edges = filters.sobel(ndvi_smooth)
        
        # Create markers
        markers = np.zeros_like(self.indices['ndvi'], dtype=np.int32)
        
        # High vegetation areas
        markers[self.indices['ndvi'] > 0.6] = 1
        # Water areas  
        markers[self.indices['ndwi'] > 0.4] = 2
        # Built-up areas
        markers[(self.indices['ndvi'] < 0.2) & (self.indices['brightness'] > 1200)] = 3
        
        # Apply watershed
        segments = segmentation.watershed(edges, markers)
        
        self.segments = segments.astype(np.uint8)
        
    def visualize_results(self, method_name="Segmentation"):
        """Visualize the segmentation results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sentinel-2 {method_name} Results', fontsize=16)
        
        # Original RGB composite
        rgb = np.stack([self.bands['red'], self.bands['green'], self.bands['blue']], axis=2)
        rgb_norm = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
        axes[0, 0].imshow(rgb_norm)
        axes[0, 0].set_title('RGB Composite')
        axes[0, 0].axis('off')
        
        # NDVI
        ndvi_plot = axes[0, 1].imshow(self.indices['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0, 1].set_title('NDVI')
        axes[0, 1].axis('off')
        plt.colorbar(ndvi_plot, ax=axes[0, 1], fraction=0.046)
        
        # NDWI
        ndwi_plot = axes[0, 2].imshow(self.indices['ndwi'], cmap='Blues', vmin=-1, vmax=1)
        axes[0, 2].set_title('NDWI') 
        axes[0, 2].axis('off')
        plt.colorbar(ndwi_plot, ax=axes[0, 2], fraction=0.046)
        
        # Segmentation results
        seg_plot = axes[1, 0].imshow(self.segments, cmap='tab10')
        axes[1, 0].set_title('Segmentation')
        axes[1, 0].axis('off')
        plt.colorbar(seg_plot, ax=axes[1, 0], fraction=0.046)
        
        # Overlay on RGB
        axes[1, 1].imshow(rgb_norm)
        axes[1, 1].imshow(self.segments, alpha=0.5, cmap='tab10')
        axes[1, 1].set_title('Overlay on RGB')
        axes[1, 1].axis('off')
        
        # Statistics
        unique, counts = np.unique(self.segments[self.segments > 0], return_counts=True)
        axes[1, 2].bar(unique, counts)
        axes[1, 2].set_title('Segment Statistics')
        axes[1, 2].set_xlabel('Segment ID')
        axes[1, 2].set_ylabel('Pixel Count')
        
        plt.tight_layout()
        plt.show()
        
    def export_results(self, output_path):
        """Export segmentation results as GeoTIFF"""
        print(f"Exporting results to {output_path}")
        
        # Update profile for output
        output_profile = self.profile.copy()
        output_profile.update({
            'dtype': 'uint8',
            'count': 1
        })
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(self.segments, 1)
            
    def get_segment_stats(self):
        """Get statistics for each segment"""
        stats = {}
        unique_segments = np.unique(self.segments[self.segments > 0])
        
        for seg_id in unique_segments:
            mask = self.segments == seg_id
            stats[f'segment_{seg_id}'] = {
                'pixel_count': np.sum(mask),
                'mean_ndvi': np.mean(self.indices['ndvi'][mask]),
                'mean_ndwi': np.mean(self.indices['ndwi'][mask]),
                'mean_brightness': np.mean(self.indices['brightness'][mask])
            }
            
        return stats


# Example usage
def run_segmentation_workflow(image_path, method='threshold'):
    """
    Run the complete segmentation workflow
    
    Args:
        image_path: Path to Sentinel-2 image
        method: 'threshold', 'kmeans', or 'edge'
    """
    
    # Initialize workflow
    segmenter = Sentinel2Segmentation(image_path)
    
    # Load and process image
    segmenter.load_image()
    segmenter.calculate_indices()
    
    # Apply segmentation method
    if method == 'threshold':
        segmenter.threshold_segmentation()
    elif method == 'kmeans':
        segmenter.kmeans_segmentation(n_clusters=6)
    elif method == 'edge':
        segmenter.edge_based_segmentation()
    
    # Visualize results
    segmenter.visualize_results(method_name=f"{method.capitalize()} Segmentation")
    
    # Get statistics
    stats = segmenter.get_segment_stats()
    print("\nSegmentation Statistics:")
    for segment, data in stats.items():
        print(f"{segment}: {data['pixel_count']} pixels, "
              f"NDVI: {data['mean_ndvi']:.3f}, "
              f"NDWI: {data['mean_ndwi']:.3f}")
    
    return segmenter

# Example usage:
segmenter = run_segmentation_workflow('/Users/devseed/Documents/repos/OBIA_workflow/data/sample.tif', method='edge')
# segmenter.export_results('segmentation_results.tif')



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage
import rasterio
from collections import defaultdict

class OBIAStatistics:
    def __init__(self, image_path, segments=None):
        """
        Initialize OBIA statistics workflow
        
        Args:
            image_path: Path to Sentinel-2 image
            segments: Segmentation array (optional)
        """
        self.image_path = image_path
        self.bands = {}
        self.indices = {}
        self.segments = segments
        self.stats_df = None
        self.objects = None
        
    def load_image(self):
        """Load Sentinel-2 image and extract bands"""
        print("Loading image for statistics...")
        
        with rasterio.open(self.image_path) as src:
            self.bands['blue'] = src.read(3).astype(np.float32)
            self.bands['green'] = src.read(2).astype(np.float32)
            self.bands['red'] = src.read(1).astype(np.float32)
            self.bands['nir'] = src.read(4).astype(np.float32)
            
    def calculate_indices(self):
        """Calculate spectral indices"""
        # NDVI
        ndvi_num = self.bands['nir'] - self.bands['red']
        ndvi_den = self.bands['nir'] + self.bands['red']
        self.indices['ndvi'] = np.divide(ndvi_num, ndvi_den, 
                                       out=np.zeros_like(ndvi_num), 
                                       where=ndvi_den!=0)
        
        # NDWI
        ndwi_num = self.bands['green'] - self.bands['nir']
        ndwi_den = self.bands['green'] + self.bands['nir']
        self.indices['ndwi'] = np.divide(ndwi_num, ndwi_den,
                                       out=np.zeros_like(ndwi_num),
                                       where=ndwi_den!=0)
        
        # Enhanced Vegetation Index (EVI)
        evi_num = 2.5 * (self.bands['nir'] - self.bands['red'])
        evi_den = self.bands['nir'] + 6 * self.bands['red'] - 7.5 * self.bands['blue'] + 1
        self.indices['evi'] = np.divide(evi_num, evi_den,
                                      out=np.zeros_like(evi_num),
                                      where=evi_den!=0)
        
    def extract_objects(self):
        """Extract individual objects from segments"""
        print("Extracting objects...")
        
        if self.segments is None:
            # Simple threshold segmentation if no segments provided
            self.indices['ndvi'] = self.indices.get('ndvi', self._calculate_ndvi())
            binary = self.indices['ndvi'] > 0.3
            binary = morphology.opening(binary, morphology.disk(3))
            self.segments = measure.label(binary)
            
        # Get object properties
        self.objects = measure.regionprops(self.segments)
        print(f"Found {len(self.objects)} objects")
        
    def _calculate_ndvi(self):
        """Helper to calculate NDVI if not already done"""
        ndvi_num = self.bands['nir'] - self.bands['red']
        ndvi_den = self.bands['nir'] + self.bands['red']
        return np.divide(ndvi_num, ndvi_den, 
                        out=np.zeros_like(ndvi_num), 
                        where=ndvi_den!=0)
        
    def calculate_spectral_stats(self):
        """Calculate spectral statistics for each object"""
        print("Calculating spectral statistics...")
        
        stats_list = []
        
        for obj in self.objects:
            if obj.area < 10:  # Skip very small objects
                continue
                
            mask = self.segments == obj.label
            stats = {'object_id': obj.label}
            
            # Basic spectral statistics
            for band_name, band_data in self.bands.items():
                values = band_data[mask]
                stats[f'{band_name}_mean'] = np.mean(values)
                stats[f'{band_name}_std'] = np.std(values)
                stats[f'{band_name}_median'] = np.median(values)
                
            # Index statistics
            for idx_name, idx_data in self.indices.items():
                values = idx_data[mask]
                stats[f'{idx_name}_mean'] = np.mean(values)
                stats[f'{idx_name}_std'] = np.std(values)
                
            stats_list.append(stats)
            
        return pd.DataFrame(stats_list)
        
    def calculate_geometric_stats(self):
        """Calculate geometric/shape statistics"""
        print("Calculating geometric statistics...")
        
        geo_stats = []
        
        for obj in self.objects:
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
        
    def calculate_texture_stats(self):
        """Calculate basic texture statistics"""
        print("Calculating texture statistics...")
        
        texture_stats = []
        
        # Use NIR band for texture analysis
        nir_band = self.bands['nir']
        
        for obj in self.objects:
            if obj.area < 10:
                continue
                
            mask = self.segments == obj.label
            
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
        
    def merge_all_stats(self):
        """Combine all statistics into single dataframe"""
        print("Merging all statistics...")
        
        spectral_df = self.calculate_spectral_stats()
        geometric_df = self.calculate_geometric_stats()
        texture_df = self.calculate_texture_stats()
        
        # Merge dataframes
        self.stats_df = spectral_df
        self.stats_df = self.stats_df.merge(geometric_df, on='object_id', how='left')
        self.stats_df = self.stats_df.merge(texture_df, on='object_id', how='left')
        
        print(f"Statistics calculated for {len(self.stats_df)} objects")
        return self.stats_df
        
    def classify_objects(self):
        """Simple rule-based classification of objects"""
        if self.stats_df is None:
            self.merge_all_stats()
            
        conditions = [
            (self.stats_df['ndvi_mean'] > 0.6) & (self.stats_df['area'] > 100),
            (self.stats_df['ndwi_mean'] > 0.3) & (self.stats_df['ndvi_mean'] < 0.2),
            (self.stats_df['ndvi_mean'] < 0.2) & (self.stats_df['red_mean'] > self.stats_df['nir_mean']),
            (self.stats_df['compactness'] < 1.5) & (self.stats_df['area'] > 500)
        ]
        
        choices = ['Dense Vegetation', 'Water', 'Built-up/Bare', 'Large Structure']
        
        self.stats_df['class'] = np.select(conditions, choices, default='Mixed/Other')
        
        return self.stats_df['class'].value_counts()
        
    def plot_statistics(self):
        """Visualize key statistics"""
        if self.stats_df is None:
            self.merge_all_stats()
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('OBIA Statistics Dashboard', fontsize=16)
        
        # NDVI vs Area
        axes[0, 0].scatter(self.stats_df['area'], self.stats_df['ndvi_mean'], alpha=0.6)
        axes[0, 0].set_xlabel('Area (pixels)')
        axes[0, 0].set_ylabel('Mean NDVI')
        axes[0, 0].set_title('NDVI vs Object Size')
        
        # Compactness distribution
        axes[0, 1].hist(self.stats_df['compactness'], bins=20, alpha=0.7)
        axes[0, 1].set_xlabel('Compactness')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Object Compactness Distribution')
        
        # Texture variance vs NDVI
        if 'texture_variance' in self.stats_df.columns:
            axes[0, 2].scatter(self.stats_df['texture_variance'], 
                             self.stats_df['ndvi_mean'], alpha=0.6)
            axes[0, 2].set_xlabel('Texture Variance')
            axes[0, 2].set_ylabel('Mean NDVI')
            axes[0, 2].set_title('Texture vs Vegetation')
        
        # Classification results
        if 'class' in self.stats_df.columns:
            class_counts = self.stats_df['class'].value_counts()
            axes[1, 0].bar(range(len(class_counts)), class_counts.values)
            axes[1, 0].set_xticks(range(len(class_counts)))
            axes[1, 0].set_xticklabels(class_counts.index, rotation=45)
            axes[1, 0].set_title('Object Classification')
        
        # Spectral signature comparison
        spectral_cols = [col for col in self.stats_df.columns if col.endswith('_mean') and 
                        col.split('_')[0] in ['blue', 'green', 'red', 'nir']]
        
        if len(spectral_cols) > 0:
            mean_signature = self.stats_df[spectral_cols].mean()
            axes[1, 1].plot(range(len(mean_signature)), mean_signature.values, 'o-')
            axes[1, 1].set_xticks(range(len(mean_signature)))
            axes[1, 1].set_xticklabels([col.replace('_mean', '') for col in spectral_cols])
            axes[1, 1].set_title('Average Spectral Signature')
        
        # Size distribution
        axes[1, 2].hist(np.log10(self.stats_df['area']), bins=20, alpha=0.7)
        axes[1, 2].set_xlabel('Log10(Area)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Object Size Distribution')
        
        plt.tight_layout()
        plt.show()
        
    def export_stats(self, output_path):
        """Export statistics to CSV"""
        if self.stats_df is None:
            self.merge_all_stats()
            
        self.stats_df.to_csv(output_path, index=False)
        print(f"Statistics exported to {output_path}")
        
    def summary_report(self):
        """Generate summary report"""
        if self.stats_df is None:
            self.merge_all_stats()
            
        print("\n=== OBIA STATISTICS SUMMARY ===")
        print(f"Total objects analyzed: {len(self.stats_df)}")
        print(f"Total area covered: {self.stats_df['area'].sum()} pixels")
        print(f"Average object size: {self.stats_df['area'].mean():.1f} pixels")
        print(f"Size range: {self.stats_df['area'].min()} - {self.stats_df['area'].max()} pixels")
        
        print(f"\nSpectral Statistics:")
        print(f"Mean NDVI: {self.stats_df['ndvi_mean'].mean():.3f} ± {self.stats_df['ndvi_mean'].std():.3f}")
        print(f"Mean NDWI: {self.stats_df['ndwi_mean'].mean():.3f} ± {self.stats_df['ndwi_mean'].std():.3f}")
        
        if 'class' in self.stats_df.columns:
            print(f"\nClassification Results:")
            for class_name, count in self.stats_df['class'].value_counts().items():
                percentage = (count / len(self.stats_df)) * 100
                print(f"  {class_name}: {count} objects ({percentage:.1f}%)")


# Main workflow function
def run_obia_statistics(image_path, segments=None):
    """
    Run complete OBIA statistics workflow
    
    Args:
        image_path: Path to Sentinel-2 image
        segments: Optional segmentation array
    """
    
    # Initialize
    stats_analyzer = OBIAStatistics(image_path, segments)
    
    # Load and process
    stats_analyzer.load_image()
    stats_analyzer.calculate_indices()
    stats_analyzer.extract_objects()
    
    # Calculate all statistics
    stats_df = stats_analyzer.merge_all_stats()
    
    # Classify objects
    class_counts = stats_analyzer.classify_objects()
    
    # Generate visualizations and summary
    stats_analyzer.plot_statistics()
    stats_analyzer.summary_report()
    
    return stats_analyzer

# Example usage:
stats = run_obia_statistics('/Users/devseed/Documents/repos/OBIA_workflow/data/ortho_subset_I.tif')
# stats.export_stats('obia_statistics.csv')