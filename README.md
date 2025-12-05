# AIML in GlobalHealth Malaria Detection using YOLOv8 Small

This project implements an end to end program that detects Malaria parasites and white blood cells in medical images using the YOLOv8 small object detection model.

## Installation & Setup

You can run the pipeline either locally or in Google Colab. Google Colab is recommended for training due to GPU availability.

### Option 1: Local Setup

1.  Install Python 3.9+
2.  Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate      # Linux/Mac
    venv\Scripts\activate         # Windows
    ```

3.  Install required packages:

    ```bash
    pip install ultralytics opencv-python numpy matplotlib tqdm
    ```

4.  (Optional) Install Jupyter Notebook if running notebooks locally:

    ```bash
    pip install notebook
    jupyter notebook
    ```

### Option 2: Google Colab Setup (Recommended)

1.  Upload the project notebooks to Google Colab.
2.  Enable GPU acceleration:
    *   `Runtime > Change runtime type > Hardware accelerator > GPU`
3.  Install dependencies within the notebook:

    ```python
    !pip install ultralytics opencv-python tqdm
    ```

4.  Mount Google Drive if you plan to store data or model weights there:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## File Structure

*   **`preprocess.ipynb`**: Data preparation
*   **`code.ipynb`**: Model training
*   **`inference.ipynb`**: Model evaluation
*   **`demo.ipynb`**: Runs demo on a single image

## Notebook Descriptions

### 1. `preprocess.ipynb` — Data Preparation

This notebook transforms raw image data files into a format optimized for YOLOv8 training.

*   **Image Analysis**: Calculates and visualizes image dimension distributions to guide resizing and tiling.
*   **Contour Cropping (`crop_contour`)**:
    *   Detects the region of interest (blood smear) using Otsu thresholding + contour detection
    *   Crops images to a bounding circle around the smear
    *   Re-maps YOLO bounding box labels to new coordinates
*   **Image Tiling (`tile_image_and_labels`)**:
    *   Splits large cropped images into fixed-size 1280×1280 tiles with overlap
    *   Adjusts bounding box coordinates tile-by-tile
*   **Dataset Splitting**: Automatically creates train, val, and test directories (default split: 70/20/10).

### 2. `code.ipynb` — Model Training

Designed to run efficiently in Google Colab.

*   **Environment Setup**: Mounts Google Drive and installs dependencies. Optionally copies data to `/content` for faster IO.
*   **Model Initialization**: Loads a pre-trained `yolov8s.pt` model.
*   **Custom Backup Callback**: A `backup_run_folder_callback` saves training weights and logs to Google Drive every 5 epochs for safety.
*   **Training Loop**: Runs YOLO training with:
    *   100 epochs
    *   Image size 1280
    *   Batch size 16
    *   AdamW optimizer

### 3. `inference.ipynb` — Evaluation & Optimization

Evaluates the trained model and optimizes thresholds for accurate parasite/WBC counting.

*   **Inference Pipeline**: Loads trained weights and performs detection on test images.
*   **Custom Metrics**:
    *   Standard YOLO metrics: Precision, Recall, mAP, IoU
    *   Average Percentage Error for parasite and WBC counts per image
*   **Threshold Optimization**:
    *   Performs confidence threshold grid search for each class
    *   Generates heatmaps of Total Percentage Error
    *   Identifies optimal thresholds that minimize counting error

### 4. `demo.ipynb` — Demo

Runs demo on a single image.

## Usage

1.  **Run `preprocess.ipynb`**: Prepare the raw dataset and generate YOLO-ready tiles.
2.  **Run `code.ipynb`**: Train the YOLOv8 model. (Use Google Colab for GPU acceleration.)
3.  **Run `inference.ipynb`**: Evaluate the model, compute metrics, and optimize confidence thresholds.
4.  **Run `demo.ipynb`**: Runs demo on a single image.

## Other Files

*   **`weights/`**: Contains the best and last weights of the trained model.
*   **`final_run/`**: Contains final run logs and graphs.
*   **`examples/`**: Contains example images and associated labels.
