# coding:gbk

            
            
import cv2
import albumentations as A
import os
import numpy as np
import random

# Define the transformations
transforms = [
    A.MotionBlur(blur_limit=(199, 599), allow_shifted=True, always_apply=True, p=0.5),
    A.Defocus(radius=(80, 100), alias_blur=(150, 300), p=0.9),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, 0.6),
        contrast_limit=(-0.3, 0.3),
        brightness_by_max=True,
        p=0.7)
]

# Define the directory containing your data
data_directory = "/data/home/.data/jiangjiewei/peimengjie/data/keratitis_data_ori/data/"

# Define the output directory
output_directory = "for_undergraduate/"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# List all labels (train, val, test)
labels = ['train', 'val', 'test']
# List all classes (other, keratitis, normal)
classes = ['other', 'keratitis', 'normal']

# Loop through each label
for label in labels:
    label_directory = os.path.join(data_directory, label)
    
    # Loop through each class
    for class_name in classes:
        class_directory = os.path.join(label_directory, class_name)
        
        # Ensure the class directory exists
        if not os.path.exists(class_directory):
            continue
        
        # Output directory has the same structure as the input directory
        out_dir = os.path.join(output_directory, label, class_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # List all image files in the class directory
        image_files = os.listdir(class_directory)
        
        # Loop through each image and apply a random transformation
        for image_file in image_files:
            # Read the image with OpenCV and convert it to RGB
            image_path = os.path.join(class_directory, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Randomly select a transformation
            chosen_transform = random.choices(transforms, weights=[0.1, 0.4, 0.5], k=1)[0]
            
            # Apply the chosen transformation
            transformed = chosen_transform(image=image)
            transformed_image = transformed["image"]

            # Concatenate the original image with the transformed image
            concatenated_image = np.concatenate((image, transformed_image), axis=1)
            
            # Save the transformed image
            output_path = os.path.join(out_dir, f"{chosen_transform.__class__.__name__}_{image_file}")
            cv2.imwrite(output_path, cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR))

