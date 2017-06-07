import tensorflow as tf
import numpy as np

from PIL import Image
from functools import reduce
from operator import mul

import vggnet

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

def rgb2grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def grayscale2rgb(grayscale):
    w, h = grayscale.shape
    rgbrep = np.empty((w, h, 3), dtype=np.float32)
    rgbrep[:, :, 2] = rgbrep[:, :, 1] = rgbrep[:, :, 0] = grayscale
    return rgbrep

def _calc_tensor_size(tensor):
    return reduce(mul, (dim.value for dim in tensor.get_shape()), 1)

def style_transfer(neural_net, content, styles, style_layer_weight_exponent, pooling, initial, initial_noiseblend):
	overall_shape = (1,) + content.shape
    per_style_shape = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_network_weights, vgg_network_mean_pixel = vggnet.load_network(neural_net)

    # layers weighted and normalised

    layer_weight = 1.0
    weights_of_style_layers = {}
    for layer in STYLE_LAYERS:
        weights_of_style_layers[layer] = layer_weight
        layer_weight *= style_layer_weight_exponent

    sum_of_layer_weights = 0
    for layer in STYLE_LAYERS:
        sum_of_layer_weights += weights_of_style_layers[layer]
    for layer in STYLE_LAYERS:
        weights_of_style_layers[layer] /= sum_of_layer_weights


    # feedforward computation of content features for CPU

    feedforward_content_graph = tf.Graph()
    with feedforward_content_graph.as_default(), feedforward_content_graph.device('/cpu:0'), tf.Session():
        image = tf.placeholder('float', shape=overall_shape)
        network = vggnet.preloaded_network(vgg_network_weights, image, pooling)
        preprocessed_content = np.array([vggnet.normalize(content, vgg_network_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = network[layer].eval(feed_dict={image: preprocessed_content})

    # feedforward computation of style features for CPU

    for i in range(len(styles)):
        feedforward_style_graph = tf.Graph()
        with feedforward_style_graph.as_default(), feedforward_style_graph.device('/cpu:0'), tf.Session():
            image = tf.placeholder('float', shape=style_shapes[i])
            network = vggnet.preloaded_network(vgg_network_weights, image, pooling)
            preprocessed_styles = np.array([vggnet.normalize(styles[i], vgg_network_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = network[layer].eval(feed_dict={image: preprocessed_styles})
                features = np.reshape(features, (-1, features.shape[3]))
                gram_matrix = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram_matrix

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vggnet.normalize(initial, vgg_network_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vggnet.preloaded_network(vgg_network_weights, image, pooling)
