""" 
This program uses the trained keras model to predict images and visualize the
output of intermediate layers.

Use the following command to execute the code
python train.py
"""

import numpy as np
import os, glob, cv2
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model, Model
from random import shuffle
import matplotlib.pyplot as plt


# Setting this system up for enabling cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def predict(model_path = 'model.h5', 
            data_path = 'dataset/test_set/', 
            image_size = (64,64,3), 
            n_images = 5, 
            layer_visu = None):
    """
    _func_: Predicts the object in the image using the trained model
            Can visualize the models if required
    Input:
      model: traned model file
      path(optional): path to the testing dataset folder
      n_images(optional): number of images to be tested
      input_size(optional) - input size of the image, defaulted to 64x64x3
      layer_visu(optional) - list of network block to visualize; eg [0,2]
    
    Return:
    """

    # get a list of all the images in the folder
    image_paths = glob.glob(data_path + '*')
    shuffle(image_paths)

    # Load the saved CNN
    model = load_model(model_path)

    # Iterate through every image
    for image_path in image_paths[:n_images]:

        # Read the image from the file
        image = cv2.imread(image_path)
        # Resize the image to what was used during training
        image_predict = cv2.resize(image,(image_size[0],image_size[1]))
        # Expand the dimensions as keras adds additional dimension while training
        image_predict = np.expand_dims(image_predict, axis=0)
        # Predict the passed image using the trained model
        model_prediction = model.predict(image_predict)

        # Discritize the prediction
        if model_prediction[0][0] == 1: predicted_label = 'DOG'
        else:predicted_label = 'CAT'
        
        image = cv2.resize(image,(500,500))
        print(predicted_label)

        # If the user wishes to visualize layers go ahead
        if layer_visu:
            visualize_activations(model, image_predict, layer_visu)
        
        # Add the prediction text to the image
        cv2.putText(image, str(predicted_label), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,0,255),4,cv2.LINE_AA)
        # Display the image
        cv2.imshow('output', image)            
        cv2.waitKey(0)
    
def visualize_activations(model, image, layer_list):
    """
    _func_: Visualize the intermediate layer outputs aka feature maps
    Input:
      model: trained model file
      image: image to visualize
      layer_list - list of network block to visualize; eg [0,2]
    
    Return:
    """
    # An empty list that stores layer name
    layer_names = []

    # Iterating through the layers and getting the names
    for layer in model.layers:
        layer_names.append(layer.name)
        print(layer.name)

    # Extract the output of every layer
    layer_outputs = [layer.output for layer in model.layers]

    # No of kernels to be displayed in a row
    kernels_per_row = 16

    # Creates a modified model that will return the layer outputs after activations
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Extracting the output of the layers 
    activations = activation_model.predict(image)

    # Now let's display our feture maps
    for nb_layer in layer_list:

        layer_activation = activations[nb_layer]
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // kernels_per_row
        display_grid = np.zeros((size * n_cols, kernels_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(kernels_per_row):
                channel_image = layer_activation[0,
                                                :, :,
                                                col * kernels_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                            row * size : (row + 1) * size] = channel_image
        
        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_names[nb_layer])
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        
    plt.show()

# Main function this is where the code begins to execute
if __name__ == "__main__":

    # Calling the prediction function
    predict(model_path = 'model.h5', 
            data_path = 'dataset/test_set/', 
            image_size = (64,64,3),
            n_images = 20,
            layer_visu = [2,3])