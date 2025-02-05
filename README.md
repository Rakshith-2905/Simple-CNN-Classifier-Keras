# Simple Classification

This is a simple classification program using Keras. The intermediate layers are visualized to understand the activation outputs.

## Features

What's all the bells and whistles this project can perform?
* Classify Cats and dogs images
* Visualize intermediate Layers


![Convolution 2D](feature_map.png)


![Test Image](dataset/test_set/cat_or_dog_2.jpg)


## Getting started

### Initial Configuration

The project is being developed in python using Keras. Install Keras by following the instructions from the link below.

### Repo
Clone the repository using.
```shell
git clone https://github.com/Rakshith-2905/.git
```

#### Dataset
A sample dataset is available in the data folder.
Any other data can be used but the datafolder should follow this file structure

    dataset/
        |train/
            |class-A/
            |class-B/
            |class-*/        
        |valid/
            |class-A/
            |class-B/
            |class-*/
        |test/
            xyz.jpg
            abc.jpg
            *.jpg
PS: * represents all the other filenames

#### Conda setup

* Visit Anaconda.com/downloads
* Select Windows
* Download the .exe installer
* Open and run the .exe installer
* Open the Anaconda Prompt and run some Python code to have a blast

Follow this link for more help
    https://problemsolvingwithpython.com/01-Orientation/01.03-Installing-Anaconda-on-Windows/

#### Tensorflow & keras setup

In Anaconda Prompt 

Change the working directory to keras-CNN-Classification

Use  ```cd..``` to move back in directory tree

Use ```cd folder_name``` to move into a folder 


After moving to the folder create a python3 virtual environment using conda, type ``y`` when prompted

```conda create -y --name tf python==3.6```


Activate the environment

``` conda activate tf ```

Install Keras and tensorflow using the following command

```conda install -c anaconda keras```

Install image operations libraries with the following commands

```conda install matplotlib```
```conda install numpy```
```conda install pillow```
```conda install -c menpo opencv```

## Usage

* To Perform basic classification run
```shell
python predict.py
```
* To Visualize the activation layers pass the layer numbers as an argument to the predict function call
```
# Calling the prediction function
    predict(model_path = 'model.h5', 
            data_path = 'dataset/test_set/', 
            image_size = (64,64,3),
            n_images = 20,
            layer_visu = [2,3])
```
* To retrain the network
```
python train.py
```

## Links

## Authors
- [Rakshith Subramanyam](https://github.com/rakshith-2905)


## Licensing

"The code in this project is licensed under MIT license."
