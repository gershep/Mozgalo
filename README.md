# Mozgalo

Unsupervised image classification using Convolutional Neural Networks (CNNs) in TensorFlow. This project was made for the 2017 Mozgalo student competition.

## Table of Contents

- [Implementation](#implementation)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Implementation

The implementation consists of three parts:

1. Creating surrogate training data
2. Trainning a CNN to discriminate between surrogate classes
3. Classification

### Creating surrogate training data

Images are rescaled to 96 × 96 pixels, augmented, and then cropped to its center 64 × 64 pixels. Each augmentation is a composition of elementary transformations:

- Rotation of the image by an angle up to 20°
- Vertical and horizontal translation by up to 20% of the patch size
- Scaling by a factor between 0.7 and 1.4
- Adding a value between −0.1 and 0.1 to the hue
- Contrast 1: raise saturation and value of all pixels to a power between 0.5 and 2, multiply these values by a factor between 0.7 and 1.4, and add to them a value between −0.1 and 0.1
- Contrast 2: multiply every RGB component of every pixel by a random factor between 0.5 and 2

The transformations are made using Python libraries *imgaug* and *numpy*.

<img width="1292" height="418" alt="Screenshot_20260201_203728 copy" src="https://github.com/user-attachments/assets/419fedd2-5430-4a0f-91a5-501097bf6072" />

### Trainning a CNN to discriminate between surrogate classes

Each of the sets of transformed image patches forms a surrogate class. We train a CNN to discriminate between these classes.

We compare two network architectures:

- The smaller network consists of two convolutional layers with 64 filters each, followed by a fully connected layer with 128 units. We can refer to this network as 64c5-64c5-128f.

- The bigger network consists of three consolutional layers with 64, 128 and 256 filters, respectively, followed by a fully connected layer with 512 units. We can refer to this network as 64c5-128c5-256c5-512f.

The last layer of each network is succeeded by a softmax layer, which serves as the network output. In all these models, all convolutional filters are connected to a 5 × 5 region of their input. 2 × 2 max-pooling is performed after the first and second convolutional layers.

### Classification

For classification, we use the $k$-means clustering method from the *sklearn* package. Before clustering, we use the *Isolation Forest* algorithm from the same package to remove the outliers.

<img width="1351" height="447" alt="Screenshot_20260201_224225" src="https://github.com/user-attachments/assets/5b6315dc-3ca3-4827-a3f6-0c2888f60dce" />

For more information, consult the [technical documentation](https://github.com/gershep/Mozgalo/blob/master/technical_documentation.pdf) (in Croatian).

## Requirements

This project is written in Python. We use the following Python packages:

- **imgaug** for image augmentation
- **SciPy** for scientific computing (contains **Matplotlib** and **NumPy**)
- **scikit-learn** for machine learning

To train CNNs, we use **TensorFlow** with an NVIDIA GPU, for which we need the following libraries:

- **CUDA Toolkit**
- **cuDNN** for deep neural networks
- **libcupti-dev**

## Usage

To run a Python script, use the command `python3 script.py`.

Here is an overview of all the scripts:

1. `variables.py` contains all the important variables. You need to provide the path to your working directory, where the subfolder `Images` contains the images you want to train on.
2. `makeAugmentations.py` creates sets of augmented images
3. `makeLabels.py` creates folders for trainning and validation and populates them
4. `big_CNN.py` and `small_CNN.py` contain the definitions for the neural networks
5. `build_image_data.py` converts image data to TFRecords file format with Example protos
6. `load_train_image.py` loads an image from TFRecords file format
7. `big_train.py` and `small_train.py` train a CNN and output a model
8. `makeTest.py` creates the testing folder and populates it
9. `big_getFeatures.py` and `small_getFeatures.py` learn image features from testing data
10. `kmeans.py` and `kmeans2.py` classify images based on these features using the $k$-means clustering method
11. `organize.py` divides the images based on the results of classification
12.  `paint.py` displays images in a grid

For more information, consult the [user documentation](https://github.com/gershep/Mozgalo/blob/master/user_documentation.pdf) (in Croatian).

## License

This project is licensed under the [MIT License](https://github.com/gershep/Mozgalo/blob/master/LICENSE).
