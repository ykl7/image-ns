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

def style_transfer(neural_net, content, styles, style_layer_weight_exponent, pooling, initial, initial_noiseblend,
    content_weight, content_weight_blend, style_blend_weights, style_weight, total_variation_weight, learning_rate,
    beta1, beta2, epsilon, preserve_colors, no_of_iterations, print_iterations=None, checkpoint_iterations=None):

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

        # content loss computations

        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

        content_loss = 0
        all_content_losses = []
        for content_layer in CONTENT_LAYERS:
            all_content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * 
                tf.nn.l2_loss(net[content_layer] - content_features[content_layer]) / content_features[content_layer].size))
        content_loss += reduce(tf.add, all_content_losses)

        # style loss computations

        style_loss = 0
        for i in range(len(styles)):
            all_style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                features = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(features), features) / size
                style_gram = style_features[i][style_layer]
                style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, all_style_losses)

        # denoising the total variation

        total_variation_y_size = _calc_tensor_size(image[:,1:,:,:])
        total_variation_x_size = _calc_tensor_size(image[:,:,1:,:])
        total_variation_loss = total_variation_weight * 2 * ((tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) / total_variation_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) / total_variation_x_size))

        overall_loss = content_loss + style_loss + total_variation_loss

        # Optimizer setup

        training_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def progress_statistics():
            stderr.write('content loss: %g\n' % content_loss.eval())
            stderr.write('style loss: %g\n' % style_loss.eval())
            stderr.write('total variation loss: %g\n' % total_variation_loss.eval())
            stderr.write('total loss: %g\n' % overall_loss.eval())

        # Optimization

        best_loss = float('inf')
        best = None
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            stderr.write('Optimization start\n')
            if (print_iterations and print_iterations != 0):
                progress_statistics()
            for i in range(no_of_iterations):
                stderr.write('Iteration %4d/%4d\n' % (i+1, no_of_iterations))
                training_step.run()

                previous_training_step = (i == no_of_iterations-1)
                if previous_training_step or (print_iterations and i%print_iterations == 0):
                    progress_statistics()
                if (checkpoint_iterations and i%checkpoint_iterations == 0) or previous_training_step:
                    present_loss = overall_loss.eval()
                    if present_loss < best_loss:
                        best_loss = present_loss
                        best = image.eval()
                    output_image = vggnet.retrieve_original(best.reshape(shape[1:]), vgg_network_mean_pixel)

                    if preserve_colors and preserve_colors == True:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(output_image, 0, 255)

                    # Luminosity transfer steps

                    # Rec.601 luma (0.299, 0.587, 0.114) conversion of stylized RGB to stylized grayscale
                    styled_grayscale = rgb2grayscale(styled_image)
                    styled_grayscale_rgb = grayscale2rgb(styled_grayscale)

                    # Stylized grayscale to YUV (YCbCr) conversion
                    styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                    # Original image to YUV (YCbCr) conversion
                    original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                    # Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                    w, h, _ = original_image.shape
                    combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                    combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                    combined_yuv[..., 1] = original_yuv[..., 1]
                    combined_yuv[..., 2] = original_yuv[..., 2]

                    # Recombined YUV image back to RGB
                    output_image = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))

                yield ((None if last_step else i), output_image)




