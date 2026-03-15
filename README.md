# Wildfire Risk Prediction using MODIS & FIRMS Data

This project builds a machine learning pipeline to detect wildfire events using **MODIS satellite data** and **NASA FIRMS fire detections**.

The pipeline processes raw satellite imagery, aligns datasets with different spatial and temporal resolutions, generates labeled wildfire training data, and trains machine learning models to predict fire occurrence.

The goal is to **identify environmental patterns associated with wildfire risk** using satellite-derived features such as vegetation indices and land surface temperature.

---

# Pipeline Overview

The full workflow consists of the following steps:

1. **Satellite Data Preprocessing**
   - Process MODIS NDVI, EVI, and Land Surface Temperature (LST) data.
   - Resolve temporal and spatial mismatches between datasets.

2. **Dataset Alignment**
   - Align MOD13Q1 vegetation indices with MOD11A2 temperature data.
   - Convert satellite data into consistent 1 km grid cells.

3. **Fire Label Generation**
   - Map NASA FIRMS wildfire detections (lat/lon points) to MODIS grid cells.

4. **Feature Dataset Creation**
   - Extract environmental features for each grid cell.
   - Label cells as **fire** or **no fire**.

5. **Model Training**
   - Train multiple machine learning models:
     - Logistic Regression
     - Random Forest
     - Gradient Boosted Trees

6. **Evaluation**
   - Evaluate models using recall, precision, F1-score, and confusion matrices.

Because wildfire detection is a **high-recall problem**, the models prioritize **detecting fires rather than minimizing false alarms**.

---

## Repository Structure

```
Wildfire-Risk-Prediction
├── notebooks
│   ├── mod11a2_preprocessing.ipynb      # Preprocessing for MOD11A2 land surface temperature data
│   ├── mod13q1_preprocessing.ipynb      # Preprocessing for MOD13Q1 vegetation index data
│   ├── utils.ipynb                      # Helper functions and utilities used across notebooks
│   ├── modis_firms_pipeline.ipynb       # Pipeline to merge MODIS and FIRMS datasets
│   │
│   ├── logistic_regression.ipynb        # Logistic Regression wildfire prediction model
│   ├── random_forest.ipynb              # Random Forest wildfire prediction model
│   └── gradient_boost.ipynb             # Gradient Boosting wildfire prediction model
│
├── data
│   ├── raw                              # Original unprocessed datasets
│   │   ├── DL_FIRE_SV-C2_658945
│   │   │   ├── fire_nrt_SV-C2_658945.csv
│   │   │   └── fire_archive_SV-C2_658945.csv
│   │   │
│   │   ├── MOD11A2_061-20250927_000049
│   │   └── MOD13Q1_061-20250926_231752
│   │
│   └── processed                        # Cleaned and model-ready datasets
│       ├── modis_firms_dataset.npz
│       └── modis_firms_train_val_test_dataset.npz
│
└── README.md                            # Project documentation
```


---

## Data Preprocessing

### MOD11A2 LST Processing (`mod11a2_preprocessing.ipynb`)

- **Goal:** Convert 8-day MOD11A2 Land Surface Temperature (LST) data to 16-day composites to match MOD13Q1 temporal resolution.  
- **Key Steps:**
  - Parse MODIS HDF filenames to extract acquisition date.  
  - Identify MOD11A2 files overlapping a MOD13Q1 16-day window.  
  - Apply scale factor (0.02) and convert Kelvin to Celsius.  
  - Replace fill values with `NaN`.  
  - Combine two 8-day LST arrays into a single 16-day composite via pixel-wise averaging.  

- **Functions:**
  - `parse_day_of_year(filepath)` -> returns datetime of MODIS file.  
  - `get_matching_mod11a2_files(mod13q1_filename, mod11a2_file_list)` -> finds overlapping MOD11A2 files.  
  - `process_single_mod11a2_file(file_path)` -> processes one LST file.  
  - `process_combined_mod11a2(mod11a2_files, file_path, target_tile)` -> produces combined day & night LST arrays.  

---

### MOD13Q1 NDVI & EVI Processing (`mod13q1_preprocessing.ipynb`)

- **Goal:** Aggregate 250 m NDVI and EVI data to 1 km resolution to spatially match MOD11A2 LST.  
- **Key Steps:**
  - Replace fill values (-3000) with `NaN`.  
  - Apply scale factor (0.0001) to convert to physical values (-1 to 1).  
  - Aggregate 4×4 blocks of 250 m pixels to 1 km resolution using mean.  

- **Functions:**
  - `aggregate_to_1km(data)` -> aggregates 4800×4800 array to 1200×1200.  
  - `process_mod13q1(file_path)` -> loads, scales, and aggregates NDVI & EVI.
    
---

### Utilities (`utils.ipynb`)

- **Purpose:** Align and combine MODIS datasets, create feature arrays, project latitude/longitude to MODIS tiles, and run exploratory data analysis (EDA).  

- **Functions:**
  - `align_datasets(mod13q1_file, mod11a2_files, mod11a2_path, target_tile)` -> aligns MOD13Q1 and MOD11A2 data.  
  - `create_features(ndvi, evi, lst_day, lst_night)` -> stacks features into a (1200×1200×4) array and returns valid pixel mask.  
  - `latlon_to_tile_pixel(lat, lon)` -> converts FIRMS lat/lon to MODIS tile indices and pixel coordinates.  
  - `run_feature_eda(X_all, y_all)` -> plots feature distributions, correlations, and fire vs non-fire differences.
  - `plot_tile_analysis(...)` -> visualizes NDVI, EVI, LST day/night, valid mask, fire mask, and FIRMS fire points.  

---

### MODIS-FIRMS Integration Pipeline (`modis_firms_pipeline.ipynb`)

- **Purpose:** Combine MODIS NDVI, EVI, LST with FIRMS fire points for a given region and time window.  
- **Steps:**
  1. Load FIRMS fire data and MODIS HDF files.  
  2. Match MOD11A2 8-day composites with MOD13Q1 16-day windows.  
  3. Extract MODIS tile IDs from filenames.  
  4. Align NDVI/EVI with LST for target tile.  
  5. Stack features and create valid pixel mask.  
  6. Filter FIRMS fire points by tile and 16-day time window.  
  7. Generate fire mask (1200×1200).  
  8. Extract training samples from valid pixels.  
  9. Visualize data using `plot_tile_analysis`.  
  10. Aggregate all features and labels for modeling.  
  11. Run EDA on final dataset.  

- **Outputs:** `X_all` (features) and `y_all` (fire labels) ready for ML models.  

---

## Machine Learning Models

### Logistic Regression (`logistic_regression.ipynb`)

- **Approach:** Simple linear baseline using a single sigmoid neuron.  
- **Steps:**
  - Load processed dataset.  
  - Standardize features with `StandardScaler`.  
  - Apply class weights for severe imbalance.  
  - Train logistic regression for 10 epochs using TensorFlow.  
  - Tune classification threshold on validation set to maximize F1 score.  
  - Evaluate on validation and test sets (confusion matrix + classification report).  

---

### Random Forest (`random_forest.ipynb`)

- **Approach:** Ensemble tree-based classifier using TensorFlow Decision Forests (TF-DF).  
- **Steps:**
  - Load dataset and apply SMOTE for oversampling minority class.  
  - Convert data to TF-DF dataset format.  
  - Train `RandomForestModel` with 300 trees, max depth 8.  
  - Evaluate on validation and test sets with custom threshold (default 0.3).  
  - Outputs confusion matrix and classification metrics.  

---

### Gradient Boosted Trees (`gradient_boost.ipynb`)

- **Approach:** Gradient Boosted Trees (TF-DF) for binary classification.  
- **Steps:**
  - Load dataset and apply SMOTE (fire class ~20%).  
  - Compute class weights and apply to TF-DF dataset via `weight` column.  
  - Train `GradientBoostedTreesModel` with 500 trees, max depth 8, binomial log-likelihood loss.  
  - Evaluate on validation and test sets with low threshold (0.3) to improve recall of rare fire events.  
  - Outputs confusion matrix and classification metrics.  

---

## Data

The project uses raw satellite data and processed datasets generated by the pipeline.

Due to file size limitations, some datasets are **not included in the repository** and must be generated locally.

---

### Raw Data

**FIRMS Fire Data**

Located in: `data/raw/DL_FIRE_SV-C2_658945/`

This directory contains NASA FIRMS wildfire detection datasets:

`fire_nrt_SV-C2_658945.csv`

`fire_archive_SV-C2_658945.csv`

These files include wildfire detections with attributes such as latitude, longitude, and acquisition date. They are used as the ground truth fire labels for the machine learning models.

**MODIS MOD11A2 (Land Surface Temperature)**

Located in: `data/raw/MOD11A2_061-20250927_000049/`

This directory contains `MOD11A2` satellite data for several MODIS tiles covering California:

| Tile ID | Area |
| :------- | :------: |
| h07v05 | Northern CA | 
| h08v05 | Central CA |
| h09v05 | Eastern CA | 
| h08v04 | Northwestern CA | 
| h09v04 | Northeastern CA | 

Each tile contains multiple files across different dates. Only the .hdf files are required for the processing pipeline.

These files contain:

- Daytime Land Surface Temperature (LST_Day)
- Nighttime Land Surface Temperature (LST_Night)

Both are provided as 8-day composites at 1 km spatial resolution.

**MODIS MOD13Q1 (Vegetation Indices)**

Located in: `data/raw/MOD13Q1_061-20250926_231752/`

This directory contains `MOD13Q1` satellite data for several MODIS tiles covering California:

| Tile ID | Area |
| :------- | :------: |
| h07v05 | Northern CA | 
| h08v05 | Central CA |
| h09v05 | Eastern CA | 
| h08v04 | Northwestern CA | 
| h09v04 | Northeastern CA | 

The dataset includes multiple files across different dates. Only the .hdf files are used in the pipeline.

These files contain:

- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)

Both are provided at 250 m spatial resolution and 16-day temporal resolution.

---

### Processed Data

Processed datasets are stored in: `data/processed/`

`modis_firms_dataset.npz`

This file contains the combined dataset created by aligning MODIS features with FIRMS wildfire detections.

Features include: NDVI, EVI, LST_Day, LST_Night

Labels: 0 = no fire, 1 = fire detected

`modis_firms_train_val_test_dataset.npz`

This dataset contains the training, validation, and test splits used for model development.

It includes: X_train, y_train, X_val, y_val, X_test, y_test, X_train_balanced, y_train_balanced

---

## Large File Notice

The processed .npz datasets are too large to store in the GitHub repository, so they are not committed to Git.

They can be regenerated locally by running the preprocessing pipeline notebooks.

---

## Usage

1. **Preprocessing**
   ```bash
   jupyter notebook mod11a2_preprocessing.ipynb
   jupyter notebook mod13q1_preprocessing.ipynb

2. **Generate features & fire labels**
   ```bash
   jupyter notebook utils.ipynb
   jupyter notebook modis_firms_pipeline.ipynb
   
3. **Train Models**
   ```bash
   jupyter notebook logistic_regression.ipynb
   jupyter notebook random_forest.ipynb
   jupyter notebook gradient_boost.ipynb

4. **Visualizations**
   ```bash
   Use `plot_tile_analysis` for per-tile feature + fire visualization.
   EDA plots and correlation heatmaps are automatically generated in `run_feature_eda`.
   
---

## Model Comparison

Three models were trained to predict wildfire events using MODIS environmental features.

| Model  | Recall | Notes |
|------|-----------|--------|
| Logistic Regression | 0.02 | Baseline linear model |
| Random Forest | 0.84 | Captures nonlinear relationships |
| Gradient Boosted Trees | 0.90 | Best overall performance |

### Key Observations

- **Logistic Regression** served as a baseline model but struggled to capture nonlinear patterns in wildfire risk.
- **Random Forest** improved performance by modeling feature interactions.
- **Gradient Boosted Trees** produced the highest recall.

Because wildfire detection prioritizes **identifying fires rather than minimizing false alarms**, models were tuned to favor **higher recall**.
