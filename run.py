import os
import math

import scipy.misc
import numpy as np

from style_transfer import style_transfer
from argparse import ArgumentParser
from PIL import Image

VGG_NETWORK_PATH = 'vgg-imagenet-19.mat'

# paper implementation arguments - default

CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TOTAL_VARIATION_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXPONENT = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
NO_OF_ITERATIONS = 1000

POOLING = 'max'

def save_image(path, image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path, quality=95)

def read_image(path):
    image = scipy.misc.imread(path).astype(np.float)
    if len(image.shape) == 2:
        # grayscale
        image = np.dstack((image,image,image))
    elif image.shape[2] == 4:
        # PNG with alpha channel
        image = image[:,:,:3]
    return image

def command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('--content', dest='content', metavar='CONTENT', required=True)
    parser.add_argument('--styles', dest='styles', nargs='+', metavar='STYLE', required=True)
    parser.add_argument('--output', dest='output', metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int, dest='no_of_iterations', metavar='NO_OF_ITERATIONS', default=NO_OF_ITERATIONS)
    parser.add_argument('--print-iterations', type=int, dest='print_iterations', metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output', dest='checkpoint_output', metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int, dest='checkpoint_iterations', metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int, dest='width', metavar='WIDTH')
    parser.add_argument('--style-scales', type=float, dest='style_scales', nargs='+', metavar='STYLE_SCALE')
    parser.add_argument('--neural-net', dest='neural_net', metavar='VGG_NETWORK_PATH', default=VGG_NETWORK_PATH)
    parser.add_argument('--content-weight-blend', type=float, dest='content_weight_blend', metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float, dest='content_weight', metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float, dest='style_weight', metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exponent', type=float, dest='style_layer_weight_exponent', metavar='STYLE_LAYER_WEIGHT_EXPONENT', default=STYLE_LAYER_WEIGHT_EXPONENT)
    parser.add_argument('--style-blend-weights', type=float, dest='style_blend_weights', nargs='+', metavar='STYLE_BLEND_WEIGHTS')
    parser.add_argument('--total_variation-weight', type=float, dest='total_variation_weight', metavar='TOTAL_VARIATION_WEIGHT', default=TOTAL_VARIATION_WEIGHT)
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float, dest='beta1', metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float, dest='beta2', metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float, dest='epsilon', metavar='EPSILON', default=EPSILON)
    parser.add_argument('--initial', dest='initial', metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float, dest='initial_noiseblend', metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action='store_true', dest='preserve_colors')
    parser.add_argument('--pooling', dest='pooling', metavar='POOLING', default=POOLING)
    return parser

def main():
    parser = command_line_arguments()
    options = parser.parse_args()
    content_image = read_image(options.content)
    style_images = [read_image(style) for style in options.styles]

    width = options.width
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) / content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape

    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if options.style_scales is not None:
            style_scale = options.style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale * target_shape[1] / style_images[i].shape[1])

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # default to equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight for weight in style_blend_weights]

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])
        # Initial guess is specified, but no noise should be blended, so no noiseblend
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 0.0
    else:
        # Random generated initial guess
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 1.0
        if options.initial_noiseblend < 1.0:
            initial = content_image

    for iteration, image in style_transfer(neural_net=options.neural_net, initial=initial, initial_noiseblend=options.initial_noiseblend,
        content=content_image, styles=style_images, preserve_colors=options.preserve_colors, no_of_iterations=options.no_of_iterations,
        content_weight=options.content_weight, content_weight_blend=options.content_weight_blend, style_weight=options.style_weight,
        style_layer_weight_exponent=options.style_layer_weight_exponent, style_blend_weights=style_blend_weights, total_variation_weight=options.total_variation_weight,
        learning_rate=options.learning_rate, beta1=options.beta1, beta2=options.beta2, epsilon=options.epsilon,
        pooling=options.pooling, print_iterations=options.print_iterations, checkpoint_iterations=options.checkpoint_iterations):
        output_file = None
        combined_rgb = image
        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
        else:
            output_file = options.output
        if output_file:
            save_image(output_file, combined_rgb)


if __name__ == '__main__':
    main()

