# AIML in GlobalHealth Malaria Detection using YOLOv8 small

This project implements a complete pipeline for detecting Malaria parasites and White Blood Cells (WBC) in medical images using the YOLOv8 small object detection model.

## File Structure

1.  **`preprocess.ipynb`**: Data preparation
2.  **`code.ipynb`**: Model training
3.  **`inference.ipynb`**: Model evaluation

## Notebook Descriptions

### 1. `preprocess.ipynb` (Data Preparation)
This notebook transforms raw image data files into a format optimized for YOLOv8 training.

*   **Image Analysis**: Calculates and visualizes the distribution of image dimensions to inform resizing and tiling strategies.
*   **Contour Cropping (`crop_contour`)**:
    *   Automatically detects the region of interest (the blood smear) using Otsu's thresholding and contour detection.
    *   Crops the image to a bounding circle around the smear to remove irrelevant background.
    *   Re-maps existing YOLO labels to the new cropped coordinates.
*   **Image Tiling (`tile_image_and_labels`)**:
    *   Splits large cropped images into smaller, fixed-size tiles (1280x1280) with overlap.
    *   Adjusts bounding box coordinates for each tile.
*   **Dataset Splitting**: Randomly partitions the processed tiles into `train`, `val`, and `test` sets (default: 70/20/10 split).

### 2. `code.ipynb` (Model Training)
This notebook handles the training of the YOLOv8 model, designed to run in a Google Colab environment.

*   **Environment Setup**: Mounts Google Drive for persistent storage and installs packages. Transfers data from Google Drive to a temporary directory for faster data access.
*   **Model Initialization**: Loads a pre-trained `yolov8s.pt` (small) model.
*   **Custom Backup Callback**:
    *   Implements a `backup_run_folder_callback` that copies training run folder (weights, logs) to Google Drive every 5 epochs.
*   **Training Loop**: Executes `model.train()` with specified hyperparameters (100 epochs, image size 1280, batch size 16, AdamW optimizer).

### 3. `inference.ipynb` (Evaluation & Optimization)
This notebook evaluates a model's performance on a test dataset and optimizes confidence thresholds for accurate object counting.

*   **Inference Pipeline**: Loads the trained weights
*   **Custom Metrics**:
    *   Calculates standard YOLO metrics: Precision, Recall, mAP, and IoU.
    *   **Average Percentage Error**: Computes the error in counting parasites and WBCs per image.
*   **Threshold Optimization**:
    *   Performs a grid search on a subset of the data over confidence thresholds for each class (parasite vs. WBC).
    *   Generates a heatmap of the Total Percentage Error to visualize how different thresholds affect counting accuracy.
    *   Identifies and applies the optimal confidence values to minimize the counting error.

## Usage

1.  **Run `preprocess.ipynb`**: Prepare your raw data. Ensure source paths are correctly set.
2.  **Run `code.ipynb`**: Train the model. This is best run on a good GPU in Google Colab.
3.  **Run `inference.ipynb`**: Evaluate the model on your local machine or Colab using the saved weights from the training step.
