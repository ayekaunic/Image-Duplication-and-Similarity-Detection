# imports
from PIL import Image
import imagehash
import os
import pandas as pd

# functions
def compute_hash(image_path):
    img = Image.open(image_path).convert('L')
    hash_value = imagehash.phash(img)
    return hash_value

def find_duplicate_images(folder_path):
    hash_dict = {}
    
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    data = {'File Name': [], 'Is Duplicate': [], 'Duplicate Of': []}

    for path in image_paths:
        hash_value = compute_hash(path)

        if hash_value in hash_dict:
            data['File Name'].append(os.path.basename(path))
            data['Is Duplicate'].append('Yes')
            data['Duplicate Of'].append(os.path.basename(hash_dict[hash_value]))
        else:
            hash_dict[hash_value] = path

            data['File Name'].append(os.path.basename(path))
            data['Is Duplicate'].append('No')
            data['Duplicate Of'].append('')

    df = pd.DataFrame(data)
    df.to_csv("report.csv", index=False)
    print("Report generated.")
    
# main
if __name__ == '__main__':
    find_duplicate_images("./images/")