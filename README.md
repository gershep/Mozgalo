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




## License

This project is licensed under the [MIT License](https://github.com/gershep/Mozgalo/blob/master/LICENSE).
