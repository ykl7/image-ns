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

def style_transfer(neural_net, content, styles, style_layer_weight_exponent):
	overall_shape = (1,) + content.shape
    per_style_shape = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_network_weights, vgg_network_mean_pixel = vgg.load_network(neural_net)

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