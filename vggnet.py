import tensorflow as tf
import numpy as np

import scipy.io

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def _convolution_layer(input, weights, bias):
    convolution = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(convolution, bias)

def load_network(path):
    data = scipy.io.loadmat(path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    return weights, mean_pixel

def normalize(image, mean_pixel):
    return image - mean_pixel

def retrieve_original(image, mean_pixel):
    return image + mean_pixel

def _pooling_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

def preloaded_network(weights, input_image, pooling):
    vggnet = {}
    current_image = input_image
    for i, name in enumerate(VGG19_LAYERS):
        # first 4 letters are constant in layer name so it can be used to check type
        layer_type = name[:4]
        if layer_type == 'conv':
            # weights as tuple of height, weight, in_channels, out_channels
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current_image = _convolution_layer(current_image, kernels, bias)
        elif layer_type == 'relu':
            current_image = tf.nn.relu(current_image)
        elif layer_type == 'pool':
            current_image = _pooling_layer(current_image, pooling)
        vggnet[name] = current_image

    assert len(vggnet) == len(VGG19_LAYERS)
    return vggnet
