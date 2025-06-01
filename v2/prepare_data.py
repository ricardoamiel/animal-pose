import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Create directories if they don't exist
for dir_name in ['train', 'val', 'target']:
    os.makedirs(os.path.join('data', dir_name), exist_ok=True)

# Read the training CSV file
df = pd.read_csv('training.csv')

# Get the list of all image files in the data directory
all_images = set(f.lstrip('0') for f in os.listdir('data') if f.endswith('.jpg'))

# Get the list of images from training.csv and strip leading zeros
training_images = set(df['Id'].astype(str).str.lstrip('0') + '.jpg')

# Split training data into train and validation sets (80-20 split)
train_images, val_images = train_test_split(
    list(training_images),
    test_size=0.2,
    random_state=42
)

# Get target images (images that are not in training set)
target_images = all_images - training_images

# Function to move files
def move_files(file_list, target_dir):
    for file_name in file_list:
        src = os.path.join('data', file_name)
        dst = os.path.join('data', target_dir, file_name)
        if os.path.exists(src):
            shutil.move(src, dst)

# Move files to their respective directories
print("Moving files to train directory...")
move_files(train_images, 'train')

print("Moving files to validation directory...")
move_files(val_images, 'val')

print("Moving files to target directory...")
move_files(target_images, 'target')

print("Data organization complete!")
