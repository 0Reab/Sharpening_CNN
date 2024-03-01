import numpy as np
from keras import layers, models
import tensorflow as tf
import os
from PIL import Image
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Sequential, Model
from keras.layers import Conv2D, Activation, BatchNormalization, Conv2DTranspose, Input, Conv2D, LeakyReLU, Add, Concatenate

# Tests if tensorflow can utilize your GPU for training
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Define the CNN model for deconvolution
def create_deconvolution_model():
    inputs = Input(shape=shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv3])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv2])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv1])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)  # Assuming output is a single-channel image

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Define the input image dimensions
image_height = 256
image_width = 256
num_channels = 1  # Assuming grayscale images

shape = (image_height, image_width, num_channels)

def load_training_data_blurred(data_dir):
    X_train = []
    for filename in os.listdir(data_dir):
        img = Image.open(os.path.join(data_dir, filename)).convert('L')  # Open image using Pillow
        img = np.array(img)  # Convert image to numpy array
        img = img / 255.0
        X_train.append(img)
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], image_height, image_width, num_channels)
    return X_train

def load_training_data_sharp(data_dir):
    Y_train = []
    for filename in os.listdir(data_dir):
        img = Image.open(os.path.join(data_dir, filename)).convert('L')  # Open image using Pillow
        img = np.array(img)  # Convert image to numpy array
        img = img / 255.0
        Y_train.append(img)
    Y_train = np.array(Y_train)
    Y_train = Y_train.reshape(Y_train.shape[0], image_height, image_width, num_channels)
    return Y_train

# Create an instance of the deconvolution model
deconvolution_model = create_deconvolution_model()

# Define the learning rate you want to use
learning_rate = 0.00005  # Example learning rate, you can adjust as needed

# Create an instance of the Adam optimizer with the desired learning rate
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with appropriate loss function and optimizer
deconvolution_model.compile(optimizer=optimizer, loss='mean_squared_error')

# Specify the directory paths where the blurred and sharp FITS images are stored
blurred_data_dir = 'blurred'
sharp_data_dir = 'sharp'

# Load the training data for blurred and sharp images
X_train = load_training_data_blurred(blurred_data_dir)
Y_train = load_training_data_sharp(sharp_data_dir)

# Train the model using the blurred and sharp astronomical FITS image pairs
deconvolution_model.fit(X_train, Y_train, epochs=45, batch_size=16)

# save model
deconvolution_model.save('deconvolution_model.h5')
deconvolution_model.save('deconvolution_model.keras')
def load_blurred_image(data_dir):
    files = os.listdir(data_dir)
    filename = files[0]
    img = Image.open(os.path.join(data_dir, filename)).convert('L')  # Open image using Pillow
    img = np.array(img)  # Convert image to numpy array
    if img is None:
        print(f"Failed to load image: {filename}")
        return None
    img = img / 255.0
    blurred_image = img.reshape(1, image_height, image_width, num_channels)
    return blurred_image

blurred_image = load_blurred_image('test')
deconvolved_image = deconvolution_model.predict(blurred_image)

# Define the output
output_file = 'deconvolved_image.jpg'
test_file = 'test_file.jpg'

# Finish code for saving deconvolved_image.jpg and test_file.jpg
# Ensure that the deconvolved and blurred images are converted to the correct data type
deconvolved_image_uint8 = (deconvolved_image[0] * 255).astype(np.uint8)
blurred_image_uint8 = (blurred_image[0] * 255).astype(np.uint8)

# Convert the numpy arrays to Pillow images and save them
deconvolved_image_pil = Image.fromarray(deconvolved_image_uint8.squeeze(), mode='L')
blurred_image_pil = Image.fromarray(blurred_image_uint8.squeeze(), mode='L')

# Save the images to the specified file paths
deconvolved_image_pil.save(output_file)
blurred_image_pil.save(test_file)

# Print a message to indicate that the deconvolved image has been saved successfully
print(f"Deconvolved image saved as {output_file}")