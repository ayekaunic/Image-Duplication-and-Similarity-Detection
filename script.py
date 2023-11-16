# imports
from PIL import Image
import imagehash
import os
import pandas as pd
from tqdm import tqdm
import numpy as np 
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity

# functions
def compute_hash(image_path):
    img = Image.open(image_path).convert('L')
    hash_value = str(imagehash.average_hash(img))
    return hash_value

def find_duplicate_images(images_directory):
    image_files = [f for f in os.listdir(images_directory) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    num_files = len(image_files)

    duplicates_matrix = pd.DataFrame(index=image_files, columns=image_files).fillna(' ')

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
    
def load_image(image_path):

    input_image = Image.open(image_path)
    resized_image = input_image.resize((224, 224))

    return resized_image

def get_image_embeddings(object_image : image):

    image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
    image_embedding = vgg16.predict(image_array, verbose=0)

    return image_embedding

def get_similarity_score(first_image_embedding, second_image_embedding):

    similarity_score = cosine_similarity(first_image_embedding, second_image_embedding).reshape(1,)
    return similarity_score

def find_similar_images(images_directory):
    image_files = [f for f in os.listdir(images_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    similarity_matrix = pd.DataFrame(index=image_files, columns=image_files)
    image_embeddings = {}

    for file in tqdm(image_files, desc="Calculating Embeddings"):
        image_path = os.path.join(images_directory, file)
        img = load_image(image_path)
        img_embedding = get_image_embeddings(img)
        image_embeddings[file] = img_embedding

    for file1 in tqdm(image_files, desc="Calculating Similarity"):
        for file2 in image_files:
            if file1 == file2:
                similarity = 0
            else:
                img_embedding1 = image_embeddings[file1]
                img_embedding2 = image_embeddings[file2]
                similarity = get_similarity_score(img_embedding1, img_embedding2)[0]
                similarity = round(similarity, 3)

            similarity_matrix.at[file1, file2] = similarity

    similarity_matrix.to_csv("similarity.csv", index=True)

# similarity model
vgg16 = VGG16(weights='imagenet', include_top=False, 
              pooling='max', input_shape=(224, 224, 3))

for model_layer in vgg16.layers:
  model_layer.trainable = False
    
# generating reports
if __name__ == '__main__':
    images_directory = "./images/"
    find_duplicate_images(images_directory)
    find_similar_images(images_directory)