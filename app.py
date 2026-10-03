import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def process_and_augment(base_input_folder, base_output_folder, base_augmentation_output_folder, subfolders, target_size=(250, 250), num_augmentations=3):
    # Create folders
    os.makedirs(base_input_folder, exist_ok=True)
    os.makedirs(base_output_folder, exist_ok=True)
    os.makedirs(base_augmentation_output_folder, exist_ok=True)

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    # 1. Processing and Resizing
    for subfolder_name in subfolders:
        input_subfolder_path = os.path.join(base_input_folder, subfolder_name)
        output_subfolder_path = os.path.join(base_output_folder, subfolder_name)
        os.makedirs(input_subfolder_path, exist_ok=True)
        os.makedirs(output_subfolder_path, exist_ok=True)

        # Create dummy image if empty
        if not os.listdir(input_subfolder_path):
            dummy_image_path = os.path.join(input_subfolder_path, f'dummy_{subfolder_name}_data.png')
            dummy_image = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
            dummy_image.save(dummy_image_path)
            print(f"Created dummy: {dummy_image_path}")

        for filename in os.listdir(input_subfolder_path):
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(input_subfolder_path, filename)
                output_path = os.path.join(output_subfolder_path, filename)
                try:
                    with Image.open(input_path) as img:
                        resized_img = img.resize(target_size)
                        resized_img.save(output_path)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # 2. Augmentations
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])

    for subfolder_name in subfolders:
        input_subfolder_path = os.path.join(base_output_folder, subfolder_name)
        output_subfolder_path = os.path.join(base_augmentation_output_folder, subfolder_name)
        os.makedirs(output_subfolder_path, exist_ok=True)

        if not os.path.exists(input_subfolder_path) or not os.listdir(input_subfolder_path):
            continue

        for filename in os.listdir(input_subfolder_path):
            if filename.lower().endswith(image_extensions):
                original_image_path = os.path.join(input_subfolder_path, filename)
                try:
                    with Image.open(original_image_path) as img:
                        for i in range(num_augmentations):
                            augmented_img = augmentation_transforms(img)
                            name, ext = os.path.splitext(filename)
                            augmented_filename = f"{name}_aug{i}{ext}"
                            augmented_img.save(os.path.join(output_subfolder_path, augmented_filename))
                except Exception as e:
                    print(f"Error augmenting {filename}: {e}")

if __name__ == '__main__':
    print("Starting processing and augmentation pipeline...")
    process_and_augment(
        base_input_folder='./animal-classification',
        base_output_folder='./outputImg',
        base_augmentation_output_folder='./augmented_outputImg',
        subfolders=['classification of animal']
    )
    print("Pipeline run finished successfully!")
