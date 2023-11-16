# Image Duplication and Similarity Detection
This repo provides serves tool for analyzing images in a specified directory. It includes functions for finding duplicate images and calculating similarity between images.

## Requirements
- Python 3.x
- Required Python Packages: `pandas1, `Pillow (PIL)`, `imagehash`, `scikit-image`, `tqdm`

## Usage
1. Ensure that you have the required Python packages installed. You can install them using the following command:
```
pip install pandas Pillow imagehash scikit-image tqdm
```
2. Place your images in the specified directory (./images/ by default).
3. Run the script:
```
python script.py
```

## Script Structure
Imports
- *os*: Provides a way of using operating system dependent functionality.
- *pandas*: Data manipulation and analysis library.
- *PIL (Pillow)*: Python Imaging Library for opening, manipulating, and saving many different image file formats.
- *imagehash*: Library for computing perceptual hash functions for images.
- *skimage*: Image processing library.
- *tqdm*: Library for adding a progress bar to loops.

Functions
1. *compute_hash(image_path)*: Computes the hash value of an image.
2. *find_duplicate_images(images_directory)*: Finds duplicate images in the specified directory and generates a CSV report (*duplicates.csv*) with the results.
3. *calculate_similarity(image1, image2)*: Calculates the structural similarity between two images.
4. *find_similar_images(images_directory)*: Calculates the similarity between all pairs of images in the specified directory and generates a CSV report (*similarity.csv*) with the results.

Generating Reports
- The script generates two reports: *duplicates.csv* containing information about duplicate images and *similarity.csv* containing information about the similarity between images.

## Note
- The script uses the default directory path *"./images/."* Make sure to update it accordingly if your images are in a different directory.

## Disclaimer
- The script assumes that images are in common formats such as PNG, JPG, JPEG, GIF, BMP.
