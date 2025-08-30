# OBIA Workflow with Sentinel-2 Imagery

This repository presents an **Object-Based Image Analysis (OBIA) workflow** developed in Python for processing and classifying Sentinel-2 satellite imagery. The work was carried out as part of the course *Application Development: Earth Observation*, within the Copernicus in Digital Earth (CDE) Master’s Programme.  

The aim is to demonstrate how OBIA concepts can be implemented in an **open-source environment**, supporting reproducibility and accessibility in Earth Observation research.

---

## Objectives

The primary objectives of this project are:

- To implement a **complete OBIA workflow** in Python.  
- To apply this workflow to **Sentinel-2 imagery**.  
- To explore the methodological steps of OBIA, including:  
  - Image preprocessing and visualization.  
  - Segmentation of image objects.  
  - Feature extraction and description of segments.  
  - Classification of objects into meaningful land cover categories. 
  - Compare different methods of segmentation and classification. 
- To provide a **didactic Jupyter Notebook** that can be reproduced and adapted for further research.  

---

## Repository Structure

- **`obia_workflow.ipynb`**  
  The central notebook containing the full OBIA workflow. It introduces the theoretical background, implements each step in Python, and discusses intermediate and final results.  

- **`obia/`**
 contains the code **`functions.py`**, which contains useful functions that support the implementation of the OBIA workflow in the notebook.


- **`data/`**  
  Contain different image samples, to test the notebook.   

- **`environment.yml`** 
  To recriate the conda environment to reproduce the workflow.

---

## Methodology and Workflow

The notebook **`obia_workflow.ipynb`** implements the following methodological steps:

1. **Setup and Libraries**  
   Importation of required libraries (`numpy`, `pandas`, `geopandas`, `rasterio`, `scikit-learn`, `matplotlib`, `nickyspatial`) and configuration of the working environment.  

2. **Data Input and Visualization**  
   Reading Sentinel-2 imagery in `.tif` format and initial exploration of the dataset.  

3. **Segmentation**  
   Application of segmentation algorithms to delineate homogeneous image objects. Segmentation parameters are adjustable to balance detail and generalization. And a comparision of different types of segmentation is possible.

4. **Feature Extraction**  
   Calculation of spectral and spatial attributes for each object, so it is possible to see the image features.

5. **Classification**  
   Training and application of different classifier methods to assign objects to predefined land cover classes.  

6. **Visualization and Export**  
   - Visualization of intermediate and final classification results within the notebook.    

---

## Acknowledgments

This work was developed as part of the **Copernicus in Digital Earth Master’s Programme**, course *Application Development: Earth Observation*.  

It builds upon and acknowledges resources from:  
- [dtiede/obia_tutorials_DT](https://github.com/dtiede/obia_tutorials_DT)  
- [kshitijrajsharma/nickyspatial](https://github.com/kshitijrajsharma/nickyspatial)  