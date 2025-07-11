# OBIA_workflow

# OBIA Workflow for Land Cover Classification Using Sentinel-2 Imagery

## Project Description

This project implements a complete Object-Based Image Analysis (OBIA) workflow in an open-source environment using Sentinel-2 satellite imagery. The workflow includes three main steps: segmentation, feature extraction and statistics, and object-based classification. The objective is to demonstrate a reproducible and effective OBIA pipeline for land cover classification using freely available data and Python libraries. The implementation relies on the `nickyspatial` library and adapts existing OBIA tutorials.

## Objectives

- Implement an OBIA workflow using Sentinel-2 imagery
- Apply image segmentation to delineate meaningful objects
- Extract spectral, spatial, and texture features from segmented objects
- Classify objects using supervised machine learning
- Evaluate and visualize classification results

## Study Area

The selected area will include diverse land cover types (e.g., coastal, agricultural, or urban regions) to test segmentation and classification performance.

## Tools & Libraries

- Python (Jupyter Notebooks)
- [`nickyspatial`]
- `rasterio`, `geopandas`, `scikit-learn`, `matplotlib`, `numpy`
- Sentinel-2 L2A imagery (downloaded from Copernicus Open Access Hub)

## Workflow Tasks and Time Allocation

| Task | Subtasks | Time (hrs) | Responsible |
|------|----------|------------|-------------|
| **1. Project Setup** | Define study area, gather imagery, install dependencies | 10 | Bruna & Beatriz |
| **2. Literature & Tutorial Review** | Study OBIA concepts, explore `nickyspatial`, identify reusable code | 10 | Bruna & Beatriz |
| **3. Image Preprocessing** | Resample, subset, select bands, cloud masking | 10 | Bruna |
| **4. Segmentation** | Apply `nickyspatial` segmentation, tune parameters | 12 | Beatriz |
| **5. Feature Extraction** | Extract NDVI, texture, geometry features | 10 | Bruna |
| **6. Feature Statistics** | Compute summary statistics, visualize features | 10 | Beatriz |
| **7. Classification** | Train/test split, train classifier (e.g., Random Forest) | 10 | Bruna |
| **8. Accuracy Assessment** | Compute confusion matrix, accuracy, kappa | 8 | Beatriz |
| **9. Visualization & Mapping** | Plot segmented objects, classification maps | 8 | Bruna |
| **10. Documentation** | Document code and workflow steps | 6 | Bruna & Beatriz |
| **11. Final Report** | Write and format final report | 6 | Bruna & Beatriz |
| **Total** |  | **160 hours** | 80h each |

## Expected Outcomes

- A complete OBIA pipeline in Python
- Segmented and classified land cover map
- Feature importance and accuracy evaluation
- Final report detailing methods, results, and discussion

## Optional Stretch Goals

- Compare two segmentation algorithms
- Use additional imagery sources (e.g., Landsat, PlanetScope)
- Explore temporal analysis with multi-date imagery
