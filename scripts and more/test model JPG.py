import numpy as np
from keras import layers, models
import tensorflow as tf
import os
from PIL import Image

# Define the input image dimensions
image_height = 512  # New height for bigger image
image_width = 512    # New width for bigger image
num_channels = 1  # Assuming grayscale images
patch_size = 256   # Patch size for processing

shape = (image_height, image_width, num_channels)

def load_blurred_image(data_dir):
    files = os.listdir(data_dir)
    filename = files[0]
    img = Image.open(os.path.join(data_dir, filename)).convert('L')  # Open image using Pillow
    img = np.array(img)  # Convert image to numpy array
    if img is None:
        print(f"Failed to load image: {filename}")
        return None
    img = img / 255.0
    return img

def process_image(image):
    # Define a function to process a single patch
    def process_patch(patch):
        # Reshape the patch for processing
        patch = patch.reshape(1, patch_size, patch_size, num_channels)
        # Perform deconvolution on the patch
        deconvolved_patch = deconvolution_model.predict(patch)
        return deconvolved_patch

    # Get dimensions of the image
    height, width = image.shape

    # Initialize an empty array to store the deconvolved image
    deconvolved_image = np.zeros_like(image)

    # Iterate over patches and process each one
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                deconvolved_patch = process_patch(patch)
                deconvolved_image[i:i+patch_size, j:j+patch_size] = deconvolved_patch.squeeze()

    return deconvolved_image

# Load the pre-trained deconvolution model
deconvolution_model = models.load_model('deconvolution_model.h5')

# Load the blurred image
blurred_image = load_blurred_image('test')

# Process the blurred image
deconvolved_image = process_image(blurred_image)

# Define the output
output_file = 'deconvolved.jpg'
test_file = 'test_file.jpg'

# Convert the numpy arrays to Pillow images and save them
deconvolved_image_uint8 = (deconvolved_image * 255).astype(np.uint8)
blurred_image_uint8 = (blurred_image * 255).astype(np.uint8)

# Convert the numpy arrays to Pillow images and save them
deconvolved_image_pil = Image.fromarray(deconvolved_image_uint8, mode='L')
blurred_image_pil = Image.fromarray(blurred_image_uint8, mode='L')

# Save the images to the specified file paths
deconvolved_image_pil.save(output_file)
blurred_image_pil.save(test_file)

# Print a message to indicate that the deconvolved image has been saved successfully
print(f"Deconvolved image saved as {output_file}")
