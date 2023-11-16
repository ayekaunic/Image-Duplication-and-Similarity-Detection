# imports
import os
import pandas as pd
from PIL import Image
import imagehash
from skimage import io, metrics
from tqdm import tqdm

# functions
def compute_hash(image_path):
    img = Image.open(image_path).convert('L')
    hash_value = str(imagehash.average_hash(img))
    return hash_value

def find_duplicate_images(images_directory):
    image_files = [f for f in os.listdir(images_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_files = len(image_files)

    duplicates_matrix = pd.DataFrame(index=image_files, columns=image_files).fillna('_')

    for i in tqdm(range(num_files), desc="Finding Duplicates"):
        for j in range(i + 1, num_files):
            file1 = image_files[i]
            file2 = image_files[j]

            hash1 = compute_hash(os.path.join(images_directory, file1))
            hash2 = compute_hash(os.path.join(images_directory, file2))

            if hash1 == hash2:
                duplicates_matrix.at[file1, file2] = 'Duplicates'
                duplicates_matrix.at[file2, file1] = 'Duplicates'

    duplicates_matrix.to_csv('duplicates.csv')
    
def calculate_similarity(image1, image2):
    if image1.shape != image2.shape:
        return 0
    return metrics.structural_similarity(image1, image2, channel_axis=2)

def find_similar_images(images_directory):
    image_files = [f for f in os.listdir(images_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    similarity_matrix = pd.DataFrame(index=image_files, columns=image_files)

    for file1 in tqdm(image_files, desc="Calculating Similarty"):
        for file2 in image_files:
            if file1 == file2:
                similarity = 0
            else:
                image1 = io.imread(os.path.join(images_directory, file1))
                image2 = io.imread(os.path.join(images_directory, file2))
                similarity = calculate_similarity(image1, image2)
                similarity = round(similarity, 3)

            similarity_matrix.at[file1, file2] = similarity

    similarity_matrix.to_csv("similarity.csv", index=True)
    
# generating reports
if __name__ == '__main__':
    images_directory = "./images/"
    find_duplicate_images(images_directory)
    find_similar_images(images_directory)