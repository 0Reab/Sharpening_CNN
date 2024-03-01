from PIL import Image
import os
import numpy as np
from scipy.signal import convolve2d

def tile_image(image_path, output_directory_sharp, output_directory_blurred, blur=False):
    # Open the image
    img = Image.open(image_path)
    img_gray = img.convert('L')  # Convert to grayscale

    # Convert image to numpy array
    img_array = np.array(img_gray)

    # Define patch size
    patch_size = 256

    # Iterate through the image and save tiles
    for i in range(0, img_array.shape[0], patch_size):
        for j in range(0, img_array.shape[1], patch_size):
            tile = img_array[i:i + patch_size, j:j + patch_size]

            # Check if the tile size is correct (256x256)
            if tile.shape == (patch_size, patch_size):
                # Create output directory if it doesn't exist
                if not os.path.exists(output_directory_sharp):
                    os.makedirs(output_directory_sharp)
                if not os.path.exists(output_directory_blurred):
                    os.makedirs(output_directory_blurred)

                # Save sharp tile
                tile_filename = f'{os.path.splitext(os.path.basename(image_path))[0]}_tile_{i}_{j}.jpg'
                tile_path_sharp = os.path.join(output_directory_sharp, tile_filename)
                sharp_tile = Image.fromarray(tile)
                sharp_tile.save(tile_path_sharp)

                # Apply slight blurring if specified
                if blur:
                    blurred_tile = convolve2d(tile, np.ones((5, 5)) / 25.0, mode='same', boundary='wrap')
                    blurred_tile_filename = f'{os.path.splitext(os.path.basename(image_path))[0]}_blurred_tile_{i}_{j}.jpg'
                    blurred_tile_path_blurred = os.path.join(output_directory_blurred, blurred_tile_filename)
                    blurred_tile_img = Image.fromarray(blurred_tile.astype(np.uint8))
                    blurred_tile_img.save(blurred_tile_path_blurred)

if __name__ == "__main__":
    # Input directory containing images
    input_directory = "not_preprocessed"

    # Output directory for sharp tiles
    output_directory_sharp = "sharp"

    # Output directory for blurred tiles
    output_directory_blurred = "blurred"

    # Get the list of image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]

    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)

        # Split the image into tiles (sharp)
        tile_image(image_path, output_directory_sharp, output_directory_blurred, blur=True)

    print("Tiles created successfully.")
