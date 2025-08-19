# 3DFace FID Pipeline

A consolidated pipeline for calculating Fréchet Inception Distance (FID) scores between enhanced, raw, and reference face datasets. This tool combines face alignment, reference subset selection, and FID calculation into a single streamlined workflow.

## Features

* **Face Alignment**: Automatically aligns all face images to CelebA format (178x218) using facial landmarks

* **Reference Subset Selection**: Optionally selects optimal reference images based on k-nearest neighbors with enhanced dataset

* **Adaptive k Selection**: Automatically increases k value until enough unique reference images are selected to match or exceed the enhanced dataset size

* **FID Calculation**: Computes FID scores between reference and both enhanced/raw datasets

* **Batch Processing**: Efficient GPU-accelerated processing with configurable batch sizes

* **Comprehensive Logging**: Detailed progress tracking and error handling

## Installation

### Prerequisites

* Python 3.10

* CUDA-compatible GPU (recommended)

* CMake (for dlib compilation)

### Setup

1. Clone or download this repository

2. Install the required dependencies:

```bash
conda create -n fid_eval python=3.10
conda activate fid_eval
pip install -r requirements.txt
```

**Note**: The requirements include PyTorch with CUDA 11.3 support. If you need a different CUDA version, modify the torch installation commands in `requirements.txt`.

### Additional Requirements

For face recognition functionality, you may need to install additional system dependencies:

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
```

**Windows:**

* Install Visual Studio Build Tools

* Install CMake

**macOS:**

```bash
brew install cmake
```

## Usage

### Basic Usage

```bash
python consolidated_fid_pipeline.py enhanced_dataset raw_dataset reference_dataset
```

### Advanced Usage

```bash
# Skip subset selection (use full reference dataset)
python consolidated_fid_pipeline.py enhanced_dataset raw_dataset reference_dataset --no-subset-selection

# Custom parameters
python consolidated_fid_pipeline.py enhanced_dataset raw_dataset reference_dataset \
    --batch-size 256 \
    --device cuda:0 \
    --k 1 \
    --num-workers 8 \
    --output-dir ./results
```

## Arguments

### Required Arguments

* `enhanced_dataset`: Path to enhanced/generated face images

* `raw_dataset`: Path to raw/original face images

* `reference_dataset`: Path to reference face images (e.g., CelebA)

### Optional Arguments

* `--batch-size`: Batch size for processing (default: 64)

* `--device`: Device to use - cuda:0, cpu, etc. (default: cuda:0)

* `--num-workers`: Number of workers for data loading (default: 4)

* `--dims`: Inception feature dimensionality (default: 2048)

* `--k`: Initial k-th nearest neighbor for selection (default: 1, will increase automatically if needed)

* `--no-subset-selection`: Skip subset selection, use full reference dataset

* `--alignment_already_done`: Indicates alignment has already been completed (will verify aligned images exist)

* `--output-size`: Output size for aligned images \[width height\] (default: 178 218)

* `--keep-jpeg`: Keep intermediate JPEG conversion folders

* `--output-dir`: Base output directory (default: parent of enhanced dataset)

## Pipeline Steps

### 1\. Face Alignment

* Converts all images to JPEG format

* Detects facial landmarks using face_recognition library

* Aligns and crops faces to CelebA format (178x218 pixels)

* Saves aligned images with `_aligned` suffix

### 2\. Reference Subset Selection (Optional)

* Extracts InceptionV3 features from reference and enhanced datasets

* Finds k-nearest neighbors between reference and enhanced images

* Selects optimal reference subset for fair comparison

* Automatically increases k value until enough unique images are selected

* Can be skipped with `--no-subset-selection` flag

### 3\. FID Calculation

* Computes FID scores between reference and enhanced datasets

* Computes FID scores between reference and raw datasets

* Reports improvement (difference between raw and enhanced FID)

## Preprocessing Steps

### Image Alignment Process

1. **Format Conversion**:

  * All input images are first converted to JPEG format

  * Images with transparency (RGBA) have white background applied

  * All images are converted to RGB color mode

2. **Face Detection and Landmark Extraction**:

  * Face landmarks are detected using the face_recognition library

  * Key facial points including eyes, nose, mouth, and eyebrows are identified

  * If no face is detected, the image is skipped

3. **Geometric Alignment**:

  * Eye centers are calculated by averaging left and right eye landmark points

  * Image is rotated to make eyes horizontal (based on angle between eyes)

  * Eyes center is positioned at approximately 51% from the top of the image

4. **Scaling and Cropping**:

  * Target eye distance is set to 80 pixels (for 178x218 output)

  * Scale factor is calculated based on original eye distance

  * Image is cropped to maintain consistent facial proportions

  * If crop extends beyond image bounds, padding is applied

5. **Resizing**:

  * Final image is resized to target dimensions (default: 178x218 pixels)

  * LANCZOS resampling is used for high-quality resizing

### Reference Subset Selection Process

1. **Feature Extraction**:

  * InceptionV3 model is loaded with pretrained weights

  * Images are loaded in batches and processed through the model

  * 2048-dimensional feature vectors are extracted from each image

2. **Normalization**:

  * Each image is normalized by subtracting mean and dividing by standard deviation

  * Normalization is applied per-image (not across dataset)

3. **Nearest Neighbor Selection**:

  * Pairwise distances are calculated between reference and enhanced image features

  * Starting with the initial k value, for each reference image, the k-th nearest enhanced image is identified

  * If the number of unique selected images is less than the enhanced dataset size:

    * k is automatically increased

    * The selection process is repeated with the new k value

    * This continues until enough unique images are selected or maximum k is reached

  * Selected images are copied to a new directory for FID calculation

  * Progress messages show the number of images selected at each k value

### FID Calculation Process

1. **Feature Extraction**:

  * Images are loaded in batches and converted to PyTorch tensors

  * Each image is normalized by subtracting mean and dividing by standard deviation

  * Images are processed through InceptionV3 model to extract features

  * Features are pooled to 2048-dimensional vectors

2. **Statistical Calculation**:

  * Mean vector (μ) is calculated across all image features in each dataset

  * Covariance matrix (Σ) is calculated for each dataset

  * Frechet distance is computed using the formula:\
    d² = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))

3. **Numerical Stability**:

  * Small epsilon (1e-6) is added to covariance matrices if needed

  * Complex components are handled by taking real part if negligible

  * Matrix square root is computed using scipy's linalg.sqrtm

## Output Structure

```
output_directory/
├── enhanced_dataset_aligned/          # Aligned enhanced images
├── raw_dataset_aligned/               # Aligned raw images
├── reference_dataset_aligned/         # Aligned reference images
└── selected_reference_subset/         # Selected reference subset (if enabled)
```

## Example Output

```
FINAL RESULTS
============================================================
FID (Reference vs Enhanced): 15.2341
FID (Reference vs Raw): 23.4567
Improvement: 8.2226

Reference subset used: ./results/selected_reference_subset
Selected 1024 images with k=3 (started with k=1, increased automatically)
```

## Supported Image Formats

* JPEG (.jpg, .jpeg)

* PNG (.png)

* BMP (.bmp)

* TIFF (.tif, .tiff)

* WebP (.webp)

* GIF (.gif)

* PPM (.ppm)

* PGM (.pgm)

## Hardware Requirements

### Minimum Requirements

* 8GB RAM

* 4GB GPU memory (for CUDA processing)

* 10GB free disk space

### Recommended Requirements

* 16GB+ RAM

* 8GB+ GPU memory

* 50GB+ free disk space (for large datasets)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch-size` parameter

2. **Face detection fails**: Ensure images contain clear, frontal faces

3. **dlib installation fails**: Install CMake and build tools

4. **Slow processing**: Use GPU with `--device cuda:0` and increase `--num-workers`

### Performance Tips

* Use GPU acceleration when available

* Increase batch size for better GPU utilization

* Use SSD storage for faster I/O

* Adjust number of workers based on CPU cores

## License

This project is provided as-is for research purposes. Please check individual component licenses for commercial use.

## Contributing

Feel free to submit issues and enhancement requests!