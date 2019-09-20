""" 
    This program trains a simple classifier in keras for the task of classifying 
    elements of the data set.
    The program requires the data tree to be of the following format.
    dataset/
        |train/
            |class-A/
            |class-B/
        
        |test/
            |class-A/
            |class-B/
            
    To train a simple classification network use the following command
    python train.py
"""

import os
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_classifier(image_shape = (64, 64, 3)):

    # Initialising a sequential CNN
    classifier = Sequential()

    # 1st Convolution layer
    no_kernel_conv1 = 32
    size_kernel_conv1 = (3,3)
    classifier.add(Conv2D(no_kernel_conv1, 
                            size_kernel_conv1, 
                            input_shape = image_shape, 
                            activation = 'relu'))

    # 1st Pooling layer
    size_kernel_pool1 = (2,2)
    classifier.add(MaxPooling2D(pool_size = size_kernel_pool1))

    # 2st Convolution layer
    no_kernel_conv2 = 32
    size_kernel_conv2 = (3,3)
    classifier.add(Conv2D(no_kernel_conv2, size_kernel_conv2, activation = 'relu'))

    # 2st Pooling layer
    size_kernel_pool2 = (2,2)
    classifier.add(MaxPooling2D(pool_size = size_kernel_pool2))

    # Flattening layer: Unwinding the pixels into a single thread
    classifier.add(Flatten())

    # Fully connected layers
    number_fc_neuron = 256
    classifier.add(Dense(units = number_fc_neuron, activation = 'relu'))

    # Final classification layer
    number_class = 1
    classifier.add(Dense(units = number_class, activation = 'sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

# Main function this is where the code begins to execute
if __name__ == "__main__":

    # Training parameters
    epoch = 5

    # Create a classifier by calling the method
    classifier = create_classifier()

    # Create a datagenerator for the training dataset
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

    # Create a datagenerator for the testing dataset
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    validation_set = validation_datagen.flow_from_directory('dataset/validation_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    # Training the classifier
    model = classifier.fit_generator(training_set,
                            steps_per_epoch = 2000,
                            epochs = 5,
                            validation_data = validation_set,
                            validation_steps = 1000)

    # Saving the trained model
    classifier.save('model.h5')

    # Save the model configuration to a text file
    with open('model_report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

