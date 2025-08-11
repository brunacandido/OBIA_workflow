import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage import measure, morphology
import rasterio
import seaborn as sns


def load_image_for_classification(image_path):
    """Load Sentinel-2 image"""
    print("Loading image for classification...")
    
    bands = {}
    with rasterio.open(image_path) as src:
        bands['blue'] = src.read(3).astype(np.float32)
        bands['green'] = src.read(2).astype(np.float32)
        bands['red'] = src.read(1).astype(np.float32)
        bands['nir'] = src.read(4).astype(np.float32)
        
    return bands


def calculate_classification_indices(bands):
    """Calculate spectral indices"""
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
    
    return indices


def prepare_classification_features(stats_df=None, segments=None, bands=None, indices=None):
    """Prepare feature matrix from statistics or direct calculation"""
    print("Preparing features...")
    
    if stats_df is not None:
        # Use pre-computed statistics
        feature_cols = [col for col in stats_df.columns 
                       if col not in ['object_id', 'class']]
        features = stats_df[feature_cols].fillna(0)
        return features, feature_cols, stats_df
        
    else:
        # Calculate features directly from segments
        if segments is None:
            # Create simple segmentation
            binary = indices['ndvi'] > 0.3
            binary = morphology.opening(binary, morphology.disk(3))
            segments = measure.label(binary)
        
        features_list = []
        objects = measure.regionprops(segments)
        
        for obj in objects:
            if obj.area < 10:
                continue
                
            mask = segments == obj.label
            
            # Basic spectral features
            feature_row = {
                'object_id': obj.label,
                'area': obj.area,
                'red_mean': np.mean(bands['red'][mask]),
                'green_mean': np.mean(bands['green'][mask]),
                'blue_mean': np.mean(bands['blue'][mask]),
                'nir_mean': np.mean(bands['nir'][mask]),
                'ndvi_mean': np.mean(indices['ndvi'][mask]),
                'ndwi_mean': np.mean(indices['ndwi'][mask]),
                'compactness': (obj.perimeter ** 2) / (4 * np.pi * obj.area),
                'eccentricity': obj.eccentricity,
                'solidity': obj.solidity
            }
            features_list.append(feature_row)
        
        stats_df = pd.DataFrame(features_list)
        feature_columns = [col for col in stats_df.columns if col != 'object_id']
        features = stats_df[feature_columns]
        
        return features, feature_columns, stats_df


def create_training_labels(stats_df):
    """Create training labels using rule-based classification"""
    print("Creating training labels...")
    
    conditions = [
        (stats_df['ndvi_mean'] > 0.5) & (stats_df['nir_mean'] > 2000),
        (stats_df['ndwi_mean'] > 0.2) & (stats_df['ndvi_mean'] < 0.2),
        (stats_df['ndvi_mean'] < 0.2) & (stats_df['red_mean'] > stats_df['nir_mean']),
        (stats_df['ndvi_mean'] > 0.2) & (stats_df['ndvi_mean'] < 0.5)
    ]
    
    choices = ['Vegetation', 'Water', 'Built-up', 'Mixed']
    
    stats_df['labels'] = np.select(conditions, choices, default='Other')
    
    # Remove 'Other' class for training (too ambiguous)
    training_data = stats_df[stats_df['labels'] != 'Other'].copy()
    
    print(f"Training samples per class:")
    print(training_data['labels'].value_counts())
    
    return training_data


def train_random_forest_classifier(features, labels):
    """Train Random Forest classifier"""
    print("Training Random Forest...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels)
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Accuracy: {accuracy:.3f}")
    
    return rf, accuracy, y_test, y_pred


def train_svm_classifier(features, labels):
    """Train SVM classifier"""
    print("Training SVM...")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SVM Accuracy: {accuracy:.3f}")
    
    return svm, scaler, accuracy, y_test, y_pred


def predict_all_objects(stats_df, feature_columns, models, scalers):
    """Predict classes for all objects using trained models"""
    print("Predicting all objects...")
    
    features = stats_df[feature_columns].fillna(0)
    predictions = {}
    
    for model_name, model in models.items():
        if scalers[model_name] is not None:
            features_scaled = scalers[model_name].transform(features)
            preds = model.predict(features_scaled)
        else:
            preds = model.predict(features)
            
        stats_df[f'{model_name}_prediction'] = preds
        predictions[model_name] = preds
        
    return predictions


def create_classification_maps(stats_df, segments, models):
    """Create classification maps"""
    if segments is None:
        print("No segments available for mapping")
        return {}
        
    classification_maps = {}
    
    for model_name in models.keys():
        class_map = np.zeros_like(segments)
        
        for idx, row in stats_df.iterrows():
            object_id = row['object_id']
            prediction = row[f'{model_name}_prediction']
            
            # Map class names to numbers for visualization
            class_mapping = {'Vegetation': 1, 'Water': 2, 'Built-up': 3, 'Mixed': 4, 'Other': 5}
            class_num = class_mapping.get(prediction, 0)
            
            class_map[segments == object_id] = class_num
            
        classification_maps[model_name] = class_map
        
    return classification_maps


def plot_classification_results(rf_results, svm_results, models, feature_columns, stats_df):
    """Visualize classification results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('OBIA Classification Results', fontsize=16)
    
    # Confusion matrices
    rf_cm = confusion_matrix(rf_results[2], rf_results[3])
    svm_cm = confusion_matrix(svm_results[3], svm_results[4])
    
    sns.heatmap(rf_cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Random Forest Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    sns.heatmap(svm_cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Greens')
    axes[0, 1].set_title('SVM Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # Feature importance (Random Forest)
    if hasattr(models['random_forest'], 'feature_importances_'):
        importances = models['random_forest'].feature_importances_
        feature_names = feature_columns[:len(importances)]
        
        # Plot top 10 features
        indices = np.argsort(importances)[::-1][:10]
        axes[0, 2].bar(range(len(indices)), importances[indices])
        axes[0, 2].set_xticks(range(len(indices)))
        axes[0, 2].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        axes[0, 2].set_title('Top 10 Feature Importances (RF)')
    
    # Classification distribution
    if 'random_forest_prediction' in stats_df.columns:
        rf_counts = stats_df['random_forest_prediction'].value_counts()
        axes[1, 0].pie(rf_counts.values, labels=rf_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Random Forest Predictions')
        
    if 'svm_prediction' in stats_df.columns:
        svm_counts = stats_df['svm_prediction'].value_counts()
        axes[1, 1].pie(svm_counts.values, labels=svm_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('SVM Predictions')
    
    # Accuracy comparison
    accuracies = [rf_results[1], svm_results[2]]
    model_names = ['Random Forest', 'SVM']
    axes[1, 2].bar(model_names, accuracies, color=['blue', 'green'], alpha=0.7)
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Model Comparison')
    axes[1, 2].set_ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        axes[1, 2].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()


def plot_classification_maps(classification_maps, bands=None):
    """Plot classification maps"""
    if not classification_maps:
        print("No classification maps available")
        return
        
    fig, axes = plt.subplots(1, len(classification_maps) + 1, figsize=(15, 5))
    
    # Original RGB
    if bands is not None:
        rgb = np.stack([bands['red'], bands['green'], bands['blue']], axis=2)
        rgb_norm = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
        axes[0].imshow(rgb_norm)
        axes[0].set_title('Original RGB')
        axes[0].axis('off')
    
    # Classification maps
    for i, (model_name, class_map) in enumerate(classification_maps.items(), 1):
        im = axes[i].imshow(class_map, cmap='tab10', vmin=0, vmax=5)
        axes[i].set_title(f'{model_name.replace("_", " ").title()} Classification')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def export_classification_results(stats_df, output_path):
    """Export classification results"""
    stats_df.to_csv(output_path, index=False)
    print(f"Classification results exported to {output_path}")


def generate_classification_summary(stats_df, models):
    """Generate classification summary"""
    print("\n=== OBIA CLASSIFICATION SUMMARY ===")
    print(f"Total objects classified: {len(stats_df)}")
    
    for model_name in models.keys():
        if f'{model_name}_prediction' in stats_df.columns:
            print(f"\n{model_name.replace('_', ' ').title()} Results:")
            counts = stats_df[f'{model_name}_prediction'].value_counts()
            for class_name, count in counts.items():
                percentage = (count / len(stats_df)) * 100
                print(f"  {class_name}: {count} objects ({percentage:.1f}%)")


def run_classification_workflow(image_path, segments=None, stats_df=None):
    """
    Run complete OBIA classification workflow
    
    Args:
        image_path: Path to Sentinel-2 image
        segments: Optional segmentation array
        stats_df: Optional pre-computed statistics
    """
    
    bands = None
    indices = None
    
    if stats_df is None:
        # Load and process if no stats provided
        bands = load_image_for_classification(image_path)
        indices = calculate_classification_indices(bands)
    
    # Prepare features and labels
    features, feature_columns, stats_df = prepare_classification_features(
        stats_df, segments, bands, indices)
    training_data = create_training_labels(stats_df)
    
    if len(training_data) < 1:
        print("Warning: Very few training samples. Results may be unreliable.")
        return stats_df, {}, {}
    
    # Train models
    train_features = training_data[feature_columns]
    train_labels = training_data['labels']
    
    rf_results = train_random_forest_classifier(train_features, train_labels)
    svm_results = train_svm_classifier(train_features, train_labels)
    
    # Store models and scalers
    models = {
        'random_forest': rf_results[0],
        'svm': svm_results[0]
    }
    scalers = {
        'random_forest': None,
        'svm': svm_results[1]
    }
    
    # Predict all objects
    predictions = predict_all_objects(stats_df, feature_columns, models, scalers)
    
    # Create classification maps
    classification_maps = create_classification_maps(stats_df, segments, models)
    
    # Visualize results
    plot_classification_results(rf_results, svm_results, models, feature_columns, stats_df)
    plot_classification_maps(classification_maps, bands)
    generate_classification_summary(stats_df, models)
    
    return stats_df, models, classification_maps


# Example usage with workflow integration:
"""
# Step 1: Segmentation (creates the segments)
from segmentation_functions import run_segmentation_workflow
bands, indices, segments, profile = run_segmentation_workflow('your_image.tif', method='threshold')

# Step 2: Statistics (uses the segments)
from statistics_functions import run_obia_statistics
stats_df, objects, segments = run_obia_statistics('your_image.tif', segments=segments)

# Step 3: Classification (uses segments + statistics)
stats_df, models, classification_maps = run_obia_classification('your_image.tif', segments=segments, stats_df=stats_df)
export_classification_results(stats_df, 'classification_results.csv')
"""

# Example usage:
# stats_df, models, classification_maps = run_obia_classification('/Users/devseed/Documents/repos/OBIA_workflow/data/ortho_subset_I.tif')
# export_classification_results(stats_df, 'classification_results.csv')