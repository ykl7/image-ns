import numpy as np

from PIL import Image

import os

VGG_PATH = 'vgg-imagenet-19.mat'

# paper implementation arguments - default

CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000

# it is seen that avg pooling turns out to be slightly better than max pooling

POOLING = 'avg'

def save_image(path, img):
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path, quality=95)

if __name__ == '__main__':
    main()