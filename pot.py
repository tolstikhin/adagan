# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class implements POT training.

"""
import collections
import logging
import os
import time
import tensorflow as tf
import utils
from utils import ProgressBar
from utils import TQDM
import numpy as np
import ops
from metrics import Metrics
slim = tf.contrib.slim


def vgg_16(inputs,
           is_training=False,
           dropout_keep_prob=0.5,
           scope='vgg_16',
           fc_conv_padding='VALID', reuse=None):
    inputs = inputs * 255.0
    inputs -= tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
      end_points_collection = sc.name + '_end_points'
      end_points = {}
      # Collect outputs for conv2d, fully_connected and max_pool2d.
      with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                          outputs_collections=end_points_collection):
        end_points['pool0'] = inputs
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        end_points['pool1'] = net
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        end_points['pool2'] = net
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        end_points['pool3'] = net
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        end_points['pool4'] = net
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        end_points['pool5'] = net
  #       # Use conv2d instead of fully_connected layers.
  #       net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
  #       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
  #                          scope='dropout6')
  #       net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
  #       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
  #                          scope='dropout7')
  #       net = slim.conv2d(net, num_classes, [1, 1],
  #                         activation_fn=None,
  #                         normalizer_fn=None,
  #                         scope='fc8')
        # Convert end_points_collection into a end_point dict.
  #       end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points

def compute_moments(_inputs, moments=[2, 3]):
    """From an image input, compute moments"""
    _inputs_sq = tf.square(_inputs)
    _inputs_cube = tf.pow(_inputs, 3)
    height = int(_inputs.get_shape()[1])
    width = int(_inputs.get_shape()[2])
    channels = int(_inputs.get_shape()[3])
    def ConvFlatten(x, kernel_size):
#                 w_sum = tf.ones([kernel_size, kernel_size, channels, 1]) / (kernel_size * kernel_size * channels)
        w_sum = tf.eye(num_rows=channels, num_columns=channels, batch_shape=[kernel_size * kernel_size])
        w_sum = tf.reshape(w_sum, [kernel_size, kernel_size, channels, channels])
        w_sum = w_sum / (kernel_size * kernel_size)
        sum_ = tf.nn.conv2d(x, w_sum, strides=[1, 1, 1, 1], padding='VALID')
        size = prod_dim(sum_)
        assert size == (height - kernel_size + 1) * (width - kernel_size + 1) * channels, size
        return tf.reshape(sum_, [-1, size])
    outputs = []
    for size in [3, 4, 5]:
        mean = ConvFlatten(_inputs, size)
        square = ConvFlatten(_inputs_sq, size)
        var = square - tf.square(mean)
        if 2 in moments:
            outputs.append(var)
        if 3 in moments:
            cube = ConvFlatten(_inputs_cube, size)
            skewness = cube - 3.0 * mean * var - tf.pow(mean, 3)  # Unnormalized
            outputs.append(skewness)
    return tf.concat(outputs, 1)

def prod_dim(tensor):
    return np.prod([int(d) for d in tensor.get_shape()[1:]])

def flatten(tensor):
    return tf.reshape(tensor, [-1, prod_dim(tensor)])

class Pot(object):
    """A base class for running individual POTs.

    """
    def __init__(self, opts, data, weights):

        # Create a new session with session.graph = default graph
        self._session = tf.Session()
        self._trained = False
        self._data = data
        self._data_weights = np.copy(weights)
        # Latent noise sampled ones to apply decoder while training
        self._noise_for_plots = opts['pot_pz_std'] * utils.generate_noise(opts, 1000)
        # Placeholders
        self._real_points_ph = None
        self._noise_ph = None
        # Init ops
        self._additional_init_ops = []
        self._init_feed_dict = {}

        # Main operations

        # Optimizers

        with self._session.as_default(), self._session.graph.as_default():
            logging.error('Building the graph...')
            self._build_model_internal(opts)

        # Make sure AdamOptimizer, if used in the Graph, is defined before
        # calling global_variables_initializer().
        init = tf.global_variables_initializer()
        self._session.run(init)
        self._session.run(self._additional_init_ops, self._init_feed_dict)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleaning the whole default Graph
        logging.error('Cleaning the graph...')
        tf.reset_default_graph()
        logging.error('Closing the session...')
        # Finishing the session
        self._session.close()

    def train(self, opts):
        """Train a POT model.

        """
        with self._session.as_default(), self._session.graph.as_default():
            self._train_internal(opts)
            self._trained = True

    def sample(self, opts, num=100):
        """Sample points from the trained POT model.

        """
        assert self._trained, 'Can not sample from the un-trained POT'
        with self._session.as_default(), self._session.graph.as_default():
            return self._sample_internal(opts, num)

    def train_mixture_discriminator(self, opts, fake_images):
        """Train classifier separating true data from points in fake_images.

        Return:
            prob_real: probabilities of the points from training data being the
                real points according to the trained mixture classifier.
                Numpy vector of shape (self._data.num_points,)
            prob_fake: probabilities of the points from fake_images being the
                real points according to the trained mixture classifier.
                Numpy vector of shape (len(fake_images),)

        """
        with self._session.as_default(), self._session.graph.as_default():
            return self._train_mixture_discriminator_internal(opts, fake_images)


    def _run_batch(self, opts, operation, placeholder, feed,
                   placeholder2=None, feed2=None):
        """Wrapper around session.run to process huge data.

        It is asumed that (a) first dimension of placeholder enumerates
        separate points, and (b) that operation is independently applied
        to every point, i.e. we can split it point-wisely and then merge
        the results. The second placeholder is meant either for is_train
        flag for batch-norm or probabilities of dropout.

        TODO: write util function which will be called both from this method
        and MNIST classification evaluation as well.

        """
        assert len(feed.shape) > 0, 'Empry feed.'
        num_points = feed.shape[0]
        batch_size = opts['tf_run_batch_size']
        batches_num = int(np.ceil((num_points + 0.) / batch_size))
        result = []
        for idx in xrange(batches_num):
            if idx == batches_num - 1:
                if feed2 is None:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:]})
                else:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:],
                                   placeholder2: feed2})
            else:
                if feed2 is None:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:
                                                     (idx + 1) * batch_size]})
                else:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:
                                                     (idx + 1) * batch_size],
                                   placeholder2: feed2})

            if len(res.shape) == 1:
                # convert (n,) vector to (n,1) array
                res = np.reshape(res, [-1, 1])
            result.append(res)
        result = np.vstack(result)
        assert len(result) == num_points
        return result

    def _build_model_internal(self, opts):
        """Build a TensorFlow graph with all the necessary ops.

        """
        assert False, 'POT base class has no build_model method defined.'

    def _train_internal(self, opts):
        assert False, 'POT base class has no train method defined.'

    def _sample_internal(self, opts, num):
        assert False, 'POT base class has no sample method defined.'

    def _train_mixture_discriminator_internal(self, opts, fake_images):
        assert False, 'POT base class has no mixture discriminator method defined.'


class ImagePot(Pot):
    """A simple POT implementation, suitable for pictures.

    """

    def __init__(self, opts, data, weights):

        # One more placeholder for batch norm
        self._is_training_ph = None
        Pot.__init__(self, opts, data, weights)


    def dcgan_like_arch(self, opts, noise, is_training, reuse, keep_prob):
        output_shape = self._data.data_shape
        num_units = opts['g_num_filters']

        batch_size = tf.shape(noise)[0]
        num_layers = opts['g_num_layers']
        if opts['g_arch'] == 'dcgan':
            height = output_shape[0] / 2**num_layers
            width = output_shape[1] / 2**num_layers
        elif opts['g_arch'] == 'dcgan_mod':
            height = output_shape[0] / 2**(num_layers-1)
            width = output_shape[1] / 2**(num_layers-1)
        else:
            assert False

        h0 = ops.linear(
            opts, noise, num_units * height * width, scope='h0_lin')
        h0 = tf.reshape(h0, [-1, height, width, num_units])
        h0 = tf.nn.relu(h0)
        layer_x = h0
        for i in xrange(num_layers-1):
            scale = 2**(i+1)
            if opts['g_stride1_deconv']:
                # Sylvain, I'm worried about this part!
                _out_shape = [batch_size, height * scale / 2,
                              width * scale / 2, num_units / scale * 2]
                layer_x = ops.deconv2d(
                    opts, layer_x, _out_shape, d_h=1, d_w=1,
                    scope='h%d_deconv_1x1' % i)
                layer_x = tf.nn.relu(layer_x)
            _out_shape = [batch_size, height * scale, width * scale, num_units / scale]
            layer_x = ops.deconv2d(opts, layer_x, _out_shape, scope='h%d_deconv' % i)
            if opts['batch_norm']:
                layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d' % i)
            layer_x = tf.nn.relu(layer_x)
            if opts['dropout']:
                _keep_prob = tf.minimum(
                    1., 0.9 - (0.9 - keep_prob) * float(i + 1) / (num_layers - 1))
                layer_x = tf.nn.dropout(layer_x, _keep_prob)

        _out_shape = [batch_size] + list(output_shape)
        if opts['g_arch'] == 'dcgan':
            last_h = ops.deconv2d(
                opts, layer_x, _out_shape, scope='hlast_deconv')
        elif opts['g_arch'] == 'dcgan_mod':
            last_h = ops.deconv2d(
                opts, layer_x, _out_shape, d_h=1, d_w=1, scope='hlast_deconv')
        else:
            assert False

        if opts['input_normalize_sym']:
            return tf.nn.tanh(last_h)
        else:
            return tf.nn.sigmoid(last_h)

    def began_dec(self, opts, noise, is_training, reuse, keep_prob):
        """ Architecture reported here: https://arxiv.org/pdf/1703.10717.pdf
        """

        output_shape = self._data.data_shape
        num_units = opts['g_num_filters']
        num_layers = opts['g_num_layers']
        batch_size = tf.shape(noise)[0]

        h0 = ops.linear(
            opts, noise, num_units * 8 * 8, scope='h0_lin')
        h0 = tf.reshape(h0, [-1, 8, 8, num_units])
        layer_x = h0
        for i in xrange(num_layers):
            if i % 3 < 2:
                # Don't change resolution
                layer_x = ops.conv2d(opts, layer_x, num_units, d_h=1, d_w=1, scope='h%d_conv' % i)
                layer_x = tf.nn.elu(layer_x)
            else:
                if i != num_layers - 1:
                    # Upsampling by factor of 2 with NN
                    scale = 2 ** (i / 3 + 1)
                    layer_x = ops.upsample_nn(layer_x, [scale * 8, scale * 8],
                                              scope='h%d_upsample' % i, reuse=reuse)
                    # Skip connection
                    append = ops.upsample_nn(h0, [scale * 8, scale * 8],
                                              scope='h%d_skipup' % i, reuse=reuse)
                    layer_x = tf.concat([layer_x, append], axis=3)

        last_h = ops.conv2d(opts, layer_x, output_shape[-1], d_h=1, d_w=1, scope='hlast_conv')

        if opts['input_normalize_sym']:
            return tf.nn.tanh(last_h)
        else:
            return tf.nn.sigmoid(last_h)

    def conv_up_res(self, opts, noise, is_training, reuse, keep_prob):
        output_shape = self._data.data_shape
        num_units = opts['g_num_filters']

        batch_size = tf.shape(noise)[0]
        num_layers = opts['g_num_layers']
        data_height = output_shape[0]
        data_width = output_shape[1]
        data_channels = output_shape[2]
        height = data_height / 2**num_layers
        width = data_width / 2**num_layers

        h0 = ops.linear(
            opts, noise, num_units * height * width, scope='h0_lin')
        h0 = tf.reshape(h0, [-1, height, width, num_units])
        h0 = tf.nn.relu(h0)
        layer_x = h0
        for i in xrange(num_layers-1):
            layer_x = tf.image.resize_nearest_neighbor(layer_x, (2 * height, 2 * width))
            layer_x = ops.conv2d(opts, layer_x, num_units / 2, d_h=1, d_w=1, scope='conv2d_%d' % i)
            height *= 2
            width *= 2
            num_units /= 2

            if opts['g_3x3_conv'] > 0:
                before = layer_x
                for j in range(opts['g_3x3_conv']):
                    layer_x = ops.conv2d(opts, layer_x, num_units, d_h=1, d_w=1,
                                         scope='conv2d_3x3_%d_%d' % (i, j),
                                         conv_filters_dim=3)
                    layer_x = tf.nn.relu(layer_x)
                layer_x += before  # Residual connection.

            if opts['batch_norm']:
                layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d' % i)
            layer_x = tf.nn.relu(layer_x)
            if opts['dropout']:
                _keep_prob = tf.minimum(
                    1., 0.9 - (0.9 - keep_prob) * float(i + 1) / (num_layers - 1))
                layer_x = tf.nn.dropout(layer_x, _keep_prob)

        layer_x = tf.image.resize_nearest_neighbor(layer_x, (2 * height, 2 * width))
        layer_x = ops.conv2d(opts, layer_x, data_channels, d_h=1, d_w=1, scope='last_conv2d_%d' % i)

        if opts['input_normalize_sym']:
            return tf.nn.tanh(layer_x)
        else:
            return tf.nn.sigmoid(layer_x)

    def ali_deconv(self, opts, noise, is_training, reuse, keep_prob):
        output_shape = self._data.data_shape

        batch_size = tf.shape(noise)[0]
        noise_size = int(noise.get_shape()[1])
        data_height = output_shape[0]
        data_width = output_shape[1]
        data_channels = output_shape[2]

        noise = tf.reshape(noise, [-1, 1, 1, noise_size])

        num_units = opts['g_num_filters']
        layer_params = []
        layer_params.append([4, 1, num_units])
        layer_params.append([4, 2, num_units / 2])
        layer_params.append([4, 1, num_units / 4])
        layer_params.append([4, 2, num_units / 8])
        layer_params.append([5, 1, num_units / 8])
        # For convolution: (n - k) / stride + 1 = s
        # For transposed: (s - 1) * stride + k = n
        layer_x = noise
        height = 1
        width = 1
        for i, (kernel, stride, channels) in enumerate(layer_params):
            height = (height - 1) * stride + kernel
            width = height
            layer_x = ops.deconv2d(
                opts, layer_x, [batch_size, height, width, channels], d_h=stride, d_w=stride,
                scope='h%d_deconv' % i, conv_filters_dim=kernel, padding='VALID')
            if opts['batch_norm']:
                layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d' % i)
            layer_x = ops.lrelu(layer_x, 0.1)
        assert height == data_height
        assert width == data_width

        # Then two 1x1 convolutions.
        layer_x = ops.conv2d(opts, layer_x, num_units / 8, d_h=1, d_w=1, scope='conv2d_1x1', conv_filters_dim=1)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bnlast')
        layer_x = ops.lrelu(layer_x, 0.1)
        layer_x = ops.conv2d(opts, layer_x, data_channels, d_h=1, d_w=1, scope='conv2d_1x1_2', conv_filters_dim=1)

        if opts['input_normalize_sym']:
            return tf.nn.tanh(layer_x)
        else:
            return tf.nn.sigmoid(layer_x)

    def generator(self, opts, noise, is_training=False, reuse=False, keep_prob=1.):
        """ Decoder actually.

        """

        output_shape = self._data.data_shape
        num_units = opts['g_num_filters']

        with tf.variable_scope("GENERATOR", reuse=reuse):
            # if not opts['convolutions']:
            if opts['g_arch'] == 'mlp':
                layer_x = noise
                for i in range(opts['g_num_layers']):
                    layer_x = ops.linear(opts, layer_x, num_units, 'h%d_lin' % i)
                    layer_x = tf.nn.relu(layer_x)
                    if opts['batch_norm']:
                        layer_x = ops.batch_norm(
                            opts, layer_x, is_training, reuse, scope='bn%d' % i)
                out = ops.linear(opts, layer_x, np.prod(output_shape), 'h%d_lin' % (i + 1))
                out = tf.reshape(out, [-1] + list(output_shape))
                if opts['input_normalize_sym']:
                    return tf.nn.tanh(out)
                else:
                    return tf.nn.sigmoid(out)
            elif opts['g_arch'] in ['dcgan', 'dcgan_mod']:
                return self.dcgan_like_arch(opts, noise, is_training, reuse, keep_prob)
            elif opts['g_arch'] == 'conv_up_res':
                return self.conv_up_res(opts, noise, is_training, reuse, keep_prob)
            elif opts['g_arch'] == 'ali':
                return self.ali_deconv(opts, noise, is_training, reuse, keep_prob)
            elif opts['g_arch'] == 'began':
                return self.began_dec(opts, noise, is_training, reuse, keep_prob)
            else:
                raise ValueError('%s unknown' % opts['g_arch'])

    def discriminator(self, opts, input_, prefix='DISCRIMINATOR', reuse=False):
        """Discriminator for the GAN objective

        """
        num_units = opts['d_num_filters']
        num_layers = opts['d_num_layers']
        nowozin_trick = opts['gan_p_trick']
        # No convolutions as GAN happens in the latent space
        with tf.variable_scope(prefix, reuse=reuse):
            hi = input_
            for i in range(num_layers):
                hi = ops.linear(opts, hi, num_units, scope='h%d_lin' % (i+1))
                hi = tf.nn.relu(hi)
            hi = ops.linear(opts, hi, 1, scope='final_lin')
        if nowozin_trick:
            # We are doing GAN between our model Qz and the true Pz.
            # We know analytical form of the true Pz.
            # The optimal discriminator for D_JS(Pz, Qz) is given by:
            # Dopt(x) = log dPz(x) - log dQz(x)
            # And we know exactly dPz(x). So add log dPz(x) explicitly 
            # to the discriminator and let it learn only the remaining
            # dQz(x) term. This appeared in the AVB paper.
            assert opts['latent_space_distr'] == 'normal'
            sigma2_p = float(opts['pot_pz_std']) ** 2
            normsq = tf.reduce_sum(tf.square(input_), 1)
            hi = hi - normsq / 2. / sigma2_p \
                    - 0.5 * tf.log(2. * np.pi) \
                    - 0.5 * opts['latent_space_dim'] * np.log(sigma2_p)
        return hi

    def pz_sampler(self, opts, input_, prefix='PZ_SAMPLER', reuse=False):
        """Transformation to be applied to the sample from Pz
        We are trying to match Qz to phi(Pz), where phi is defined by
        this function
        """
        dim = opts['latent_space_dim']
        with tf.variable_scope(prefix, reuse=reuse):
            matrix = tf.get_variable(
                "W", [dim, dim], tf.float32,
                tf.constant_initializer(np.identity(dim)))
            bias = tf.get_variable(
                "b", [dim],
                initializer=tf.constant_initializer(0.))
        return tf.matmul(input_, matrix) + bias

    def get_batch_size(self, opts, input_):
        return tf.cast(tf.shape(input_)[0], tf.float32)# opts['batch_size']

    def moments_stats(self, opts, input_):
        """
        Compute estimates of the first 4 moments of the coordinates
        based on the sample in input_. Compare them to the desired
        population values and return a corresponding loss.
        """
        input_ = input_ / float(opts['pot_pz_std'])
        # If Pz = Qz then input_ should now come from 
        # a product of pz_dim Gaussians N(0, 1)
        # Thus first moments should be 0
        p1 = tf.reduce_mean(input_, 0)
        center_inp = input_ - p1 # Broadcasting
        # Second centered and normalized moments should be 1
        p2 = tf.sqrt(1e-5 + tf.reduce_mean(tf.square(center_inp), 0))
        normed_inp = center_inp / p2
        # Third central moment should be 0
        # p3 = tf.pow(1e-5 + tf.abs(tf.reduce_mean(tf.pow(center_inp, 3), 0)), 1.0 / 3.0)
        p3 = tf.abs(tf.reduce_mean(tf.pow(center_inp, 3), 0))
        # 4th central moment of any uni-variate Gaussian = 3 * sigma^4
        # p4 = tf.pow(1e-5 + tf.reduce_mean(tf.pow(center_inp, 4), 0) / 3.0, 1.0 / 4.0)
        p4 = tf.reduce_mean(tf.pow(center_inp, 4), 0) / 3.
        def zero_t(v):
            return tf.sqrt(1e-5 + tf.reduce_mean(tf.square(v)))
        def one_t(v):
            # The function below takes its minimum value 1. at v = 1.
            return tf.sqrt(1e-5 + tf.reduce_mean(tf.maximum(tf.square(v), 1.0 / (1e-5 + tf.square(v)))))
        return tf.stack([zero_t(p1), one_t(p2), zero_t(p3), one_t(p4)])

    def discriminator_test(self, opts, input_):
        """Deterministic discriminator using simple tests."""
        if opts['z_test'] == 'cramer':
            test_v = self.discriminator_cramer_test(opts, input_)
        elif opts['z_test'] == 'anderson':
            test_v = self.discriminator_anderson_test(opts, input_)
        elif opts['z_test'] == 'moments':
            test_v = tf.reduce_mean(self.moments_stats(opts, input_)) / 10.0
        elif opts['z_test'] == 'lks':
            test_v = self.discriminator_lks_test(opts, input_)
        else:
            raise ValueError('%s Unknown' % opts['z_test'])
        return test_v

    def discriminator_cramer_test(self, opts, input_):
        """Deterministic discriminator using Cramer von Mises Test.

        """
        add_dim = opts['z_test_proj_dim']
        if add_dim > 0:
            dim = int(input_.get_shape()[1])
            proj = np.random.rand(dim, add_dim)
            proj = proj - np.mean(proj, 0)
            norms = np.sqrt(np.sum(np.square(proj), 0) + 1e-5)
            proj = tf.constant(proj / norms, dtype=tf.float32)
            projected_x = tf.matmul(input_, proj)  # Shape [batch_size, add_dim].

            # Shape [batch_size, z_dim+add_dim]
            all_dims_x = tf.concat([input_, projected_x], 1)
        else:
            all_dims_x = input_

        # top_k can only sort on the last dimension and we want to sort the
        # first one (batch_size).
        batch_size = self.get_batch_size(opts, all_dims_x)
        transposed = tf.transpose(all_dims_x, perm=[1, 0])
        values, indices = tf.nn.top_k(transposed, k=tf.cast(batch_size, tf.int32))
        values = tf.reverse(values, [1])
        #values = tf.Print(values, [values], "sorted values")
        normal_dist = tf.contrib.distributions.Normal(0., float(opts['pot_pz_std']))
        #
        normal_cdf = normal_dist.cdf(values)
        #normal_cdf = tf.Print(normal_cdf, [normal_cdf], "normal_cdf")
        expected = (2 * tf.range(1, batch_size+1, 1, dtype="float") - 1) / (2.0 * batch_size)
        #expected = tf.Print(expected, [expected], "expected")
        # We don't use the constant.
        # constant = 1.0 / (12.0 * batch_size * batch_size)
        # stat = constant + tf.reduce_sum(tf.square(expected - normal_cdf), 1) / batch_size
        stat = tf.reduce_sum(tf.square(expected - normal_cdf), 1) / batch_size
        stat = tf.reduce_mean(stat)
        #stat = tf.Print(stat, [stat], "stat")
        return stat

    def discriminator_anderson_test(self, opts, input_):
        """Deterministic discriminator using the Anderson Darling test.

        """
        # A-D test says to normalize data before computing the statistic
        # Because true mean and variance are known, we are supposed to use
        # the population parameters for that, but wiki says it's better to
        # still use the sample estimates while normalizing
        means = tf.reduce_mean(input_, 0)
        input_ = input_ - means # Broadcasting
        stds = tf.sqrt(1e-5 + tf.reduce_mean(tf.square(input_), 0))
        input_= input_ / stds
        # top_k can only sort on the last dimension and we want to sort the
        # first one (batch_size).
        batch_size = self.get_batch_size(opts, input_)
        transposed = tf.transpose(input_, perm=[1, 0])
        values, indices = tf.nn.top_k(transposed, k=tf.cast(batch_size, tf.int32))
        values = tf.reverse(values, [1])
        normal_dist = tf.contrib.distributions.Normal(0., float(opts['pot_pz_std']))
        normal_cdf = normal_dist.cdf(values)
        # ln_normal_cdf is of shape (z_dim, batch_size)
        ln_normal_cdf = tf.log(normal_cdf)
        ln_one_normal_cdf = tf.log(1.0 - normal_cdf)
        w1 = 2 * tf.range(1, batch_size + 1, 1, dtype="float") - 1
        w2 = 2 * tf.range(batch_size - 1, -1, -1, dtype="float") + 1
        stat = -batch_size - tf.reduce_sum(w1 * ln_normal_cdf + \
                                           w2 * ln_one_normal_cdf, 1) / batch_size
        # stat is of shape (z_dim)
        stat = tf.reduce_mean(tf.square(stat))
        return stat

    def discriminator_lks_lin_test(self, opts, input_):
        """Deterministic discriminator using Kernel Stein Discrepancy test
        refer to LKS test on page 3 of https://arxiv.org/pdf/1705.07673.pdf

        The statistic basically reads:
            \[
                \frac{2}{n}\sum_{i=1}^n \left(
                    frac{<x_{2i}, x_{2i - 1}>}{\sigma_p^4}
                    + d/\sigma_k^2
                    - \|x_{2i} - x_{2i - 1}\|^2\left(\frac{1}{\sigma_p^2\sigma_k^2} + \frac{1}{\sigma_k^4}\right)
                \right)
                \exp( - \|x_{2i} - x_{2i - 1}\|^2/2/\sigma_k^2)
            \]

        """
        # To check the typical sizes of the test for Pz = Qz, uncomment
        # input_ = opts['pot_pz_std'] * utils.generate_noise(opts, 100000)
        batch_size = self.get_batch_size(opts, input_)
        batch_size = tf.cast(batch_size, tf.int32)
        half_size = batch_size / 2
        # s1 = tf.slice(input_, [0, 0], [half_size, -1])
        # s2 = tf.slice(input_, [half_size, 0], [half_size, -1])
        s1 = input_[:half_size, :]
        s2 = input_[half_size:, :]
        dotprods = tf.reduce_sum(tf.multiply(s1, s2), axis=1)
        distances = tf.reduce_sum(tf.square(s1 - s2), axis=1)
        sigma2_p = opts['pot_pz_std'] ** 2 # var = std ** 2
        # Median heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(distances, half_size).values[half_size - 1]
        # Maximum heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(distances, 1).values[0]
        sigma2_k = opts['latent_space_dim'] * sigma2_p
        if opts['verbose'] == 2:
            sigma2_k = tf.Print(sigma2_k, [tf.nn.top_k(distances, 1).values[0]],
                                'Maximal squared pairwise distance:')
            sigma2_k = tf.Print(sigma2_k, [tf.reduce_mean(distances)],
                                'Average squared pairwise distance:')
            sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
        # sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
        res = dotprods / sigma2_p ** 2 \
              - distances * (1. / sigma2_p / sigma2_k + 1. / sigma2_k ** 2) \
              + opts['latent_space_dim'] / sigma2_k
        res = tf.multiply(res, tf.exp(- distances / 2./ sigma2_k))
        stat = tf.reduce_mean(res)
        return stat


    def discriminator_lks_test(self, opts, input_):
        """Deterministic discriminator using Kernel Stein Discrepancy test
        refer to the quadratic test of https://arxiv.org/pdf/1705.07673.pdf

        The statistic basically reads:
            \[
                \frac{1}{n^2 - n}\sum_{i \neq j} \left(
                    frac{<x_i, x__j>}{\sigma_p^4}
                    + d/\sigma_k^2
                    - \|x_i - x_j\|^2\left(\frac{1}{\sigma_p^2\sigma_k^2} + \frac{1}{\sigma_k^4}\right)
                \right)
                \exp( - \|x_i - x_j\|^2/2/\sigma_k^2)
            \]

        """
        n = self.get_batch_size(opts, input_)
        n = tf.cast(n, tf.int32)
        half_size = (n * n - n) / 2
        nf = tf.cast(n, tf.float32)
        norms = tf.reduce_sum(tf.square(input_), axis=1, keep_dims=True)
        dotprods = tf.matmul(input_, input_, transpose_b=True)
        distances = norms + tf.transpose(norms) - 2. * dotprods
        sigma2_p = opts['pot_pz_std'] ** 2 # var = std ** 2
        # Median heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        # Maximal heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
        sigma2_k = opts['latent_space_dim'] * sigma2_p
        if opts['verbose'] == 2:
            sigma2_k = tf.Print(sigma2_k, [tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]],
                                'Maximal squared pairwise distance:')
            sigma2_k = tf.Print(sigma2_k, [tf.reduce_mean(distances)],
                                'Average squared pairwise distance:')
            sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
        res = dotprods / sigma2_p ** 2 \
              - distances * (1. / sigma2_p / sigma2_k + 1. / sigma2_k ** 2) \
              + opts['latent_space_dim'] / sigma2_k
        res = tf.multiply(res, tf.exp(- distances / 2./ sigma2_k))
        res = tf.multiply(res, 1. - tf.eye(n))
        stat = tf.reduce_sum(res) / (nf * nf - nf)
        # stat = tf.reduce_sum(res) / (nf * nf)
        return stat

    def discriminator_mmd_test(self, opts, sample_qz, sample_pz):
        """U statistic for MMD(Qz, Pz) with the RBF kernel

        """
        sigma2_p = opts['pot_pz_std'] ** 2 # var = std ** 2
        kernel = 'IM'
        n = self.get_batch_size(opts, sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2
        # Pz
        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz
        # Qz
        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz
        # Pz vs Qz
        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods


        if opts['verbose'] == 2:
            distances = tf.Print(distances, [tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]],
                                'Maximal Qz squared pairwise distance:')
            distances = tf.Print(distances, [tf.reduce_mean(distances_qz)],
                                'Average Qz squared pairwise distance:')

            distances = tf.Print(distances, [tf.nn.top_k(tf.reshape(distances_pz, [-1]), 1).values[0]],
                                'Maximal Pz squared pairwise distance:')
            distances = tf.Print(distances, [tf.reduce_mean(distances_pz)],
                                'Average Pz squared pairwise distance:')

            distances = tf.Print(distances, [tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]],
                                'Maximal squared pairwise distance:')
            distances = tf.Print(distances, [tf.nn.top_k(tf.reshape(distances, [-1]), n * n).values[n * n - 1]],
                                'Minimal squared pairwise distance:')
            distances = tf.Print(distances, [tf.reduce_mean(distances)],
                                'Average squared pairwise distance:')

        if kernel == 'RBF':
            # RBF kernel

            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
            sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp( - distances_qz / 2. / sigma2_k)
            res1 += tf.exp( - distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp( - distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
            # stat = tf.reduce_sum(res) / (nf * nf)
        elif kernel == 'IM':
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            C = 2 * opts['latent_space_dim'] * sigma2_p
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res1 = tf.Print(res1, [res1], 'First two terms:')
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            res2 = tf.Print(res2, [res2], 'Negative term:')
            stat = res1 - res2
            # stat = tf.reduce_sum(res) / (nf * nf)
        return stat

    def correlation_loss(self, opts, input_):
        """
        Independence test based on Pearson's correlation.
        Keep in mind that this captures only linear dependancies.
        However, for multivariate Gaussian independence is equivalent
        to zero correlation.
        """

        batch_size = self.get_batch_size(opts, input_)
        dim = int(input_.get_shape()[1])
        transposed = tf.transpose(input_, perm=[1, 0])
        mean = tf.reshape(tf.reduce_mean(transposed, axis=1), [-1, 1])
        centered_transposed = transposed - mean # Broadcasting mean
        cov = tf.matmul(centered_transposed, centered_transposed, transpose_b=True)
        cov = cov / (batch_size - 1)
        #cov = tf.Print(cov, [cov], "cov")
        sigmas = tf.sqrt(tf.diag_part(cov) + 1e-5)
        #sigmas = tf.Print(sigmas, [sigmas], "sigmas")
        sigmas = tf.reshape(sigmas, [1, -1])
        sigmas = tf.matmul(sigmas, sigmas, transpose_a=True)
        #sigmas = tf.Print(sigmas, [sigmas], "sigmas")
        # Pearson's correlation
        corr = cov / sigmas
        triangle = tf.matrix_set_diag(tf.matrix_band_part(corr, 0, -1), tf.zeros(dim))
        #triangle = tf.Print(triangle, [triangle], "triangle")
        loss = tf.reduce_sum(tf.square(triangle)) / ((dim * dim - dim) / 2.0)
        #loss = tf.Print(loss, [loss], "Correlation loss")
        return loss


    def encoder(self, opts, input_, is_training=False, reuse=False, keep_prob=1.):
        if opts['e_add_noise']:
            def add_noise(x):
                shape = tf.shape(x)
                return x + tf.truncated_normal(shape, 0.0, 0.01)
            def do_nothing(x):
                return x
            input_ = tf.cond(is_training, lambda: add_noise(input_), lambda: do_nothing(input_))
        num_units = opts['e_num_filters']
        num_layers = opts['e_num_layers']
        with tf.variable_scope("ENCODER", reuse=reuse):
            if not opts['convolutions']:
                hi = input_
                for i in range(num_layers):
                    hi = ops.linear(opts, hi, num_units, scope='h%d_lin' % i)
                    if opts['batch_norm']:
                        hi = ops.batch_norm(opts, hi, is_training, reuse, scope='bn%d' % i)
                    hi = tf.nn.relu(hi)
                if opts['e_is_random']:
                    latent_mean = ops.linear(
                        opts, hi, opts['latent_space_dim'], 'h%d_lin' % (i + 1))
                    log_latent_sigmas = ops.linear(
                        opts, hi, opts['latent_space_dim'], 'h%d_lin_sigma' % (i + 1))
                    return latent_mean, log_latent_sigmas
                else:
                    return ops.linear(opts, hi, opts['latent_space_dim'], 'h%d_lin' % (i + 1))
            elif opts['e_arch'] == 'dcgan':
                return self.dcgan_encoder(opts, input_, is_training, reuse, keep_prob)
            elif opts['e_arch'] == 'ali':
                return self.ali_encoder(opts, input_, is_training, reuse, keep_prob)
            elif opts['e_arch'] == 'began':
                return self.began_encoder(opts, input_, is_training, reuse, keep_prob)
            else:
                raise ValueError('%s Unknown' % opts['e_arch'])

    def dcgan_encoder(self, opts, input_, is_training=False, reuse=False, keep_prob=1.):
        num_units = opts['e_num_filters']
        num_layers = opts['e_num_layers']
        layer_x = input_
        for i in xrange(num_layers):
            scale = 2**(num_layers-i-1)
            layer_x = ops.conv2d(opts, layer_x, num_units / scale, scope='h%d_conv' % i)

            if opts['batch_norm']:
                layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d' % i)
            layer_x = tf.nn.relu(layer_x)
            if opts['dropout']:
                _keep_prob = tf.minimum(
                    1., 0.9 - (0.9 - keep_prob) * float(i + 1) / num_layers)
                layer_x = tf.nn.dropout(layer_x, _keep_prob)

            if opts['e_3x3_conv'] > 0:
                before = layer_x
                for j in range(opts['e_3x3_conv']):
                    layer_x = ops.conv2d(opts, layer_x, num_units / scale, d_h=1, d_w=1,
                                         scope='conv2d_3x3_%d_%d' % (i, j),
                                         conv_filters_dim=3)
                    layer_x = tf.nn.relu(layer_x)
                layer_x += before  # Residual connection.

        if opts['e_is_random']:
            latent_mean = ops.linear(
                opts, layer_x, opts['latent_space_dim'], scope='hlast_lin')
            log_latent_sigmas = ops.linear(
                opts, layer_x, opts['latent_space_dim'], scope='hlast_lin_sigma')
            return latent_mean, log_latent_sigmas
        else:
            return ops.linear(opts, layer_x, opts['latent_space_dim'], scope='hlast_lin')

    def ali_encoder(self, opts, input_, is_training=False, reuse=False, keep_prob=1.):
        num_units = opts['e_num_filters']
        layer_params = []
        layer_params.append([5, 1, num_units / 8])
        layer_params.append([4, 2, num_units / 4])
        layer_params.append([4, 1, num_units / 2])
        layer_params.append([4, 2, num_units])
        layer_params.append([4, 1, num_units * 2])
        # For convolution: (n - k) / stride + 1 = s
        # For transposed: (s - 1) * stride + k = n
        layer_x = input_
        height = int(layer_x.get_shape()[1])
        width = int(layer_x.get_shape()[2])
        assert height == width
        for i, (kernel, stride, channels) in enumerate(layer_params):
            height = (height - kernel) / stride + 1
            width = height
            # print((height, width))
            layer_x = ops.conv2d(
                opts, layer_x, channels, d_h=stride, d_w=stride,
                scope='h%d_conv' % i, conv_filters_dim=kernel, padding='VALID')
            if opts['batch_norm']:
                layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d' % i)
            layer_x = ops.lrelu(layer_x, 0.1)
        assert height == 1
        assert width == 1

        # Then two 1x1 convolutions.
        layer_x = ops.conv2d(opts, layer_x, num_units * 2, d_h=1, d_w=1, scope='conv2d_1x1', conv_filters_dim=1)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bnlast')
        layer_x = ops.lrelu(layer_x, 0.1)
        layer_x = ops.conv2d(opts, layer_x, num_units / 2, d_h=1, d_w=1, scope='conv2d_1x1_2', conv_filters_dim=1)

        if opts['e_is_random']:
            latent_mean = ops.linear(
                opts, layer_x, opts['latent_space_dim'], scope='hlast_lin')
            log_latent_sigmas = ops.linear(
                opts, layer_x, opts['latent_space_dim'], scope='hlast_lin_sigma')
            return latent_mean, log_latent_sigmas
        else:
            return ops.linear(opts, layer_x, opts['latent_space_dim'], scope='hlast_lin')

    def began_encoder(self, opts, input_, is_training=False, reuse=False, keep_prob=1.):
        num_units = opts['e_num_filters']
        assert num_units == opts['g_num_filters'], 'BEGAN requires same number of filters in encoder and decoder'
        num_layers = opts['e_num_layers']
        layer_x = ops.conv2d(opts, input_, num_units, scope='h_first_conv')
        for i in xrange(num_layers):
            if i % 3 < 2:
                if i != num_layers - 2:
                    ii = i - (i / 3)
                    scale = (ii + 1 - ii / 2)
                else:
                    ii = i - (i / 3)
                    scale = (ii - (ii - 1) / 2)
                layer_x = ops.conv2d(opts, layer_x, num_units * scale, d_h=1, d_w=1, scope='h%d_conv' % i)
                layer_x = tf.nn.elu(layer_x)
            else:
                if i != num_layers - 1:
                    layer_x = ops.downsample(layer_x, scope='h%d_maxpool' % i, reuse=reuse)
        # Tensor should be [N, 8, 8, filters] right now

        if opts['e_is_random']:
            latent_mean = ops.linear(
                opts, layer_x, opts['latent_space_dim'], scope='hlast_lin')
            log_latent_sigmas = ops.linear(
                opts, layer_x, opts['latent_space_dim'], scope='hlast_lin_sigma')
            return latent_mean, log_latent_sigmas
        else:
            return ops.linear(opts, layer_x, opts['latent_space_dim'], scope='hlast_lin')

    def _data_augmentation(self, opts, real_points, is_training):
        if not opts['data_augm']:
            return real_points

        height = int(real_points.get_shape()[1])
        width = int(real_points.get_shape()[2])
        depth = int(real_points.get_shape()[3])
        # logging.error("real_points shape", real_points.get_shape())
        def _distort_func(image):
            # tf.image.per_image_standardization(image), should we?
            # Pad with zeros.
            image = tf.image.resize_image_with_crop_or_pad(
                image, height+4, width+4)
            image = tf.random_crop(image, [height, width, depth])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.minimum(tf.maximum(image, 0.0), 1.0)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.3)
            image = tf.minimum(tf.maximum(image, 0.0), 1.0)
            image = tf.image.random_hue(image, 0.08)
            image = tf.minimum(tf.maximum(image, 0.0), 1.0)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.3)
            image = tf.minimum(tf.maximum(image, 0.0), 1.0)
            return image

        def _regular_func(image):
            # tf.image.per_image_standardization(image)?
            return image

        distorted_images = tf.cond(
            is_training,
            lambda: tf.map_fn(_distort_func, real_points,
                              parallel_iterations=100),
            lambda: tf.map_fn(_regular_func, real_points,
                              parallel_iterations=100))

        return distorted_images

    def _recon_loss_using_disc_encoder(
            self, opts, reconstructed_training, encoded_training,
            real_points, is_training_ph, keep_prob_ph):
        """Build an additional loss using the encoder as discriminator."""
        reconstructed_reencoded_sg = self.encoder(
            opts, tf.stop_gradient(reconstructed_training),
            is_training=is_training_ph, keep_prob=keep_prob_ph, reuse=True)
        if opts['e_is_random']:
            reconstructed_reencoded_sg = reconstructed_reencoded_sg[0]
        reconstructed_reencoded = self.encoder(
            opts, reconstructed_training, is_training=is_training_ph,
            keep_prob=keep_prob_ph, reuse=True)
        if opts['e_is_random']:
            reconstructed_reencoded = reconstructed_reencoded[0]
        # Below line enforces the forward to be reconstructed_reencoded and backwards to NOT change the encoder....
        crazy_hack = reconstructed_reencoded - reconstructed_reencoded_sg +\
            tf.stop_gradient(reconstructed_reencoded_sg)
        encoded_training_sg = self.encoder(
            opts, tf.stop_gradient(real_points),
            is_training=is_training_ph, keep_prob=keep_prob_ph, reuse=True)
        if opts['e_is_random']:
            encoded_training_sg = encoded_training_sg[0]

        adv_fake_layer = ops.linear(opts, reconstructed_reencoded_sg, 1, scope='adv_layer')
        adv_true_layer = ops.linear(opts, encoded_training_sg, 1, scope='adv_layer', reuse=True)
        adv_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=adv_fake_layer, labels=tf.zeros_like(adv_fake_layer))
        adv_true = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=adv_true_layer, labels=tf.ones_like(adv_true_layer))
        adv_fake = tf.reduce_mean(adv_fake)
        adv_true = tf.reduce_mean(adv_true)
        adv_c_loss = adv_fake + adv_true
        emb_c = tf.reduce_sum(tf.square(crazy_hack - tf.stop_gradient(encoded_training)), 1)
        emb_c_loss = tf.reduce_mean(tf.sqrt(emb_c + 1e-5))
        # Normalize the loss, so that it does not depend on how good the
        # discriminator is.
        emb_c_loss = emb_c_loss / tf.stop_gradient(emb_c_loss)
        return adv_c_loss, emb_c_loss

    def _recon_loss_using_disc_conv(self, opts, reconstructed_training, real_points, is_training, keep_prob):
        """Build an additional loss using a discriminator in X space."""
        def _conv_flatten(x, kernel_size):
            height = int(x.get_shape()[1])
            width = int(x.get_shape()[2])
            channels = int(x.get_shape()[3])
            w_sum = tf.eye(num_rows=channels, num_columns=channels, batch_shape=[kernel_size * kernel_size])
            w_sum = tf.reshape(w_sum, [kernel_size, kernel_size, channels, channels])
            w_sum = w_sum / (kernel_size * kernel_size)
            sum_ = tf.nn.conv2d(x, w_sum, strides=[1, 1, 1, 1], padding='SAME')
            size = prod_dim(sum_)
            assert size == height * width * channels, size
            return tf.reshape(sum_, [-1, size])

        def _gram_scores(tensor, kernel_size):
            assert len(tensor.get_shape()) == 4, tensor
            ttensor = tf.transpose(tensor, [3, 1, 2, 0])
            rand_indices = tf.random_shuffle(tf.range(ttensor.get_shape()[0]))
            shuffled = tf.gather(ttensor, rand_indices)

            shuffled = tf.transpose(shuffled, [3, 1, 2, 0])
            cross_p = _conv_flatten(tensor * shuffled, kernel_size)  # shape [batch_size, height * width * channels]
            diag_p = _conv_flatten(tf.square(tensor), kernel_size)  # shape [batch_size, height * width * channels]
            return cross_p, diag_p

        def _architecture(inputs, reuse=None):
            with tf.variable_scope('DISC_X_LOSS', reuse=reuse):
                num_units = opts['adv_c_num_units']
                num_layers = 1
                filter_sizes = opts['adv_c_patches_size']
                if isinstance(filter_sizes, int):
                    filter_sizes = [filter_sizes]
                else:
                    filter_sizes = [int(n) for n in filter_sizes.split(',')]
                embedded_outputs = []
                linear_outputs = []
                for filter_size in filter_sizes:
                    layer_x = inputs
                    for i in xrange(num_layers):
    #                     scale = 2**(num_layers-i-1)
                        layer_x = ops.conv2d(opts, layer_x, num_units, d_h=1, d_w=1, scope='h%d_conv%d' % (i, filter_size),
                                             conv_filters_dim=filter_size, padding='SAME')
    #                     if opts['batch_norm']:
    #                         layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d_%d' % (i, filter_size))
                        layer_x = ops.lrelu(layer_x, 0.1)
                    last = ops.conv2d(
                        opts, layer_x, 1, d_h=1, d_w=1, scope="last_lin%d" % filter_size, conv_filters_dim=1, l2_norm=True)
                    if opts['cross_p_w'] > 0.0 or opts['diag_p_w'] > 0.0:
                        cross_p, diag_p = _gram_scores(layer_x, filter_size)
                        embedded_outputs.append(cross_p * opts['cross_p_w'])
                        embedded_outputs.append(diag_p * opts['diag_p_w'])
                    fl = flatten(layer_x)
#                     fl = tf.Print(fl, [fl], "fl")
                    embedded_outputs.append(fl)
                    size = int(last.get_shape()[1])
                    linear_outputs.append(tf.reshape(last, [-1, size * size]))
                if len(embedded_outputs) > 1:
                    embedded_outputs = tf.concat(embedded_outputs, 1)
                else:
                    embedded_outputs = embedded_outputs[0]
                if len(linear_outputs) > 1:
                    linear_outputs = tf.concat(linear_outputs, 1)
                else:
                    linear_outputs = linear_outputs[0]

                return embedded_outputs, linear_outputs


        reconstructed_embed_sg, adv_fake_layer = _architecture(tf.stop_gradient(reconstructed_training), reuse=None)
        reconstructed_embed, _ = _architecture(reconstructed_training, reuse=True)
        # Below line enforces the forward to be reconstructed_embed and backwards to NOT change the discriminator....
        crazy_hack = reconstructed_embed-reconstructed_embed_sg+tf.stop_gradient(reconstructed_embed_sg)
        real_p_embed_sg, adv_true_layer = _architecture(tf.stop_gradient(real_points), reuse=True)
        real_p_embed, _ = _architecture(real_points, reuse=True)

        adv_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=adv_fake_layer, labels=tf.zeros_like(adv_fake_layer))
        adv_true = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=adv_true_layer, labels=tf.ones_like(adv_true_layer))
        adv_fake = tf.reduce_mean(adv_fake)
        adv_true = tf.reduce_mean(adv_true)

        adv_c_loss = adv_fake + adv_true
        emb_c = tf.reduce_mean(tf.square(crazy_hack - tf.stop_gradient(real_p_embed)), 1)

        real_points_shuffle = tf.stop_gradient(tf.random_shuffle(real_p_embed))
        emb_c_shuffle = tf.reduce_mean(tf.square(real_points_shuffle - tf.stop_gradient(reconstructed_embed)), 1)

        raw_emb_c_loss = tf.reduce_mean(emb_c)
        shuffled_emb_c_loss = tf.reduce_mean(emb_c_shuffle)
        emb_c_loss = raw_emb_c_loss / shuffled_emb_c_loss
        emb_c_loss = emb_c_loss * 40

        return adv_c_loss, emb_c_loss

    def _recon_loss_using_disc_conv_eb(self, opts, reconstructed_training, real_points, is_training, keep_prob):
        """Build an additional loss using a discriminator in X space, using Energy Based approach."""
        def copy3D(height, width, channels):
            m = np.zeros([height, width, channels, height, width, channels])
            for i in xrange(height):
                for j in xrange(width):
                    for c in xrange(channels):
                        m[i, j, c, i, j, c] = 1.0
            return tf.constant(np.reshape(m, [height, width, channels, -1]), dtype=tf.float32)

        def _architecture(inputs, reuse=None):
            dim = opts['adv_c_patches_size']
            height = int(inputs.get_shape()[1])
            width = int(inputs.get_shape()[2])
            channels = int(inputs.get_shape()[3])
            with tf.variable_scope('DISC_X_LOSS', reuse=reuse):
                num_units = opts['adv_c_num_units']
                num_layers = 1
                layer_x = inputs
                for i in xrange(num_layers):
#                     scale = 2**(num_layers-i-1)
                    layer_x = ops.conv2d(opts, layer_x, num_units, d_h=1, d_w=1, scope='h%d_conv' % i,
                                         conv_filters_dim=dim, padding='SAME')
#                     if opts['batch_norm']:
#                         layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d' % i)
                    layer_x = ops.lrelu(layer_x, 0.1)  #tf.nn.relu(layer_x)

                copy_w = copy3D(dim, dim, channels)
                duplicated = tf.nn.conv2d(inputs, copy_w, strides=[1, 1, 1, 1], padding='SAME')
                decoded = ops.conv2d(
                    opts, layer_x, channels * dim * dim, d_h=1, d_w=1, scope="decoder",
                    conv_filters_dim=1, padding='SAME')
            reconstruction = tf.reduce_mean(tf.square(tf.stop_gradient(duplicated) - decoded), [1, 2, 3])
            assert len(reconstruction.get_shape()) == 1
            return flatten(layer_x), reconstruction


        reconstructed_embed_sg, adv_fake_layer = _architecture(tf.stop_gradient(reconstructed_training), reuse=None)
        reconstructed_embed, _ = _architecture(reconstructed_training, reuse=True)
        # Below line enforces the forward to be reconstructed_embed and backwards to NOT change the discriminator....
        crazy_hack = reconstructed_embed-reconstructed_embed_sg+tf.stop_gradient(reconstructed_embed_sg)
        real_p_embed_sg, adv_true_layer = _architecture(tf.stop_gradient(real_points), reuse=True)
        real_p_embed, _ = _architecture(real_points, reuse=True)

        adv_fake = tf.reduce_mean(adv_fake_layer)
        adv_true = tf.reduce_mean(adv_true_layer)

        adv_c_loss = tf.log(adv_true) - tf.log(adv_fake)
        emb_c = tf.reduce_sum(tf.square(crazy_hack - tf.stop_gradient(real_p_embed)), 1)
        emb_c_loss = tf.reduce_mean(emb_c)

        return adv_c_loss, emb_c_loss

    def _recon_loss_using_vgg(self, opts, reconstructed_training, real_points, is_training, keep_prob):
        """Build an additional loss using a pretrained VGG in X space."""

        def _architecture(_inputs, reuse=None):
            _, end_points = vgg_16(_inputs, is_training=is_training, dropout_keep_prob=keep_prob, reuse=reuse)
            layer_name = opts['vgg_layer']
            if layer_name == 'concat':
                outputs = []
                for ln in ['pool1', 'pool2', 'pool3']:
                    output = end_points[ln]
                    output = flatten(output)
                    outputs.append(output)
                output = tf.concat(outputs, 1)
            elif layer_name.startswith('concat_w'):
                weights = layer_name.split(',')[1:]
                assert len(weights) == 5
                outputs = []
                for lnum in range(5):
                    num = lnum + 1
                    ln = 'pool%d' % num
                    output = end_points[ln]
                    output = flatten(output)
                    # We sqrt the weight here because we use L2 after.
                    outputs.append(np.sqrt(float(weights[lnum])) * output)
                output = tf.concat(outputs, 1)
            else:
                output = end_points[layer_name]
                output = flatten(output)
            if reuse is None:
                variables_to_restore = slim.get_variables_to_restore(include=['vgg_16'])
                path = os.path.join(opts['data_dir'], 'vgg_16.ckpt')
#                 '/tmpp/models/vgg_16.ckpt'
                init_assign_op, init_feed_dict = slim.assign_from_checkpoint(path, variables_to_restore)
                self._additional_init_ops += [init_assign_op]
                self._init_feed_dict.update(init_feed_dict)
            return output


        reconstructed_embed_sg = _architecture(tf.stop_gradient(reconstructed_training), reuse=None)
        reconstructed_embed = _architecture(reconstructed_training, reuse=True)
        # Below line enforces the forward to be reconstructed_embed and backwards to NOT change the discriminator....
        crazy_hack = reconstructed_embed-reconstructed_embed_sg+tf.stop_gradient(reconstructed_embed_sg)
        real_p_embed = _architecture(real_points, reuse=True)

        emb_c = tf.reduce_mean(tf.square(crazy_hack - tf.stop_gradient(real_p_embed)), 1)
        emb_c_loss = tf.reduce_mean(tf.sqrt(emb_c + 1e-5))
#         emb_c_loss = tf.Print(emb_c_loss, [emb_c_loss], "emb_c_loss")
#         # Normalize the loss, so that it does not depend on how good the
#         # discriminator is.
#         emb_c_loss = emb_c_loss / tf.stop_gradient(emb_c_loss)
        return emb_c_loss

    def _recon_loss_using_moments(self, opts, reconstructed_training, real_points, is_training, keep_prob):
        """Build an additional loss using moments."""

        def _architecture(_inputs):
            return compute_moments(_inputs, moments=[2])  # TODO

        reconstructed_embed = _architecture(reconstructed_training)
        real_p_embed = _architecture(real_points)

        emb_c = tf.reduce_mean(tf.square(reconstructed_embed - tf.stop_gradient(real_p_embed)), 1)
#         emb_c = tf.Print(emb_c, [emb_c], "emb_c")
        emb_c_loss = tf.reduce_mean(emb_c)
        return emb_c_loss * 100.0 * 100.0 # TODO: constant.

    def _recon_loss_using_vgg_moments(self, opts, reconstructed_training, real_points, is_training, keep_prob):
        """Build an additional loss using a pretrained VGG in X space."""

        def _architecture(_inputs, reuse=None):
            _, end_points = vgg_16(_inputs, is_training=is_training, dropout_keep_prob=keep_prob, reuse=reuse)
            layer_name = opts['vgg_layer']
            output = end_points[layer_name]
#             output = flatten(output)
            output /= 255.0  # the vgg_16 method scales everything by 255.0, so we divide back here.
            variances = compute_moments(output, moments=[2])

            if reuse is None:
                variables_to_restore = slim.get_variables_to_restore(include=['vgg_16'])
                path = os.path.join(opts['data_dir'], 'vgg_16.ckpt')
#                 '/tmpp/models/vgg_16.ckpt'
                init_assign_op, init_feed_dict = slim.assign_from_checkpoint(path, variables_to_restore)
                self._additional_init_ops += [init_assign_op]
                self._init_feed_dict.update(init_feed_dict)
            return variances

        reconstructed_embed_sg = _architecture(tf.stop_gradient(reconstructed_training), reuse=None)
        reconstructed_embed = _architecture(reconstructed_training, reuse=True)
        # Below line enforces the forward to be reconstructed_embed and backwards to NOT change the discriminator....
        crazy_hack = reconstructed_embed-reconstructed_embed_sg+tf.stop_gradient(reconstructed_embed_sg)
        real_p_embed = _architecture(real_points, reuse=True)

        emb_c = tf.reduce_mean(tf.square(crazy_hack - tf.stop_gradient(real_p_embed)), 1)
        emb_c_loss = tf.reduce_mean(emb_c)
#         emb_c_loss = tf.Print(emb_c_loss, [emb_c_loss], "emb_c_loss")
#         # Normalize the loss, so that it does not depend on how good the
#         # discriminator is.
#         emb_c_loss = emb_c_loss / tf.stop_gradient(emb_c_loss)
        return emb_c_loss   # TODO: constant.

    def add_least_gaussian2d_ops(self, opts):
        """ Add ops searching for the 2d plane in z_dim hidden space
            corresponding to the 'least Gaussian' look of the sample
        """

        with tf.variable_scope('leastGaussian2d'):
            # Projection matrix which we are going to tune
            sample_ph = tf.placeholder(
                tf.float32, [None, opts['latent_space_dim']],
                name='sample_ph')
            v = tf.get_variable(
                "proj_v", [opts['latent_space_dim'], 1],
                tf.float32, tf.random_normal_initializer(stddev=1.))
            u = tf.get_variable(
                "proj_u", [opts['latent_space_dim'], 1],
                tf.float32, tf.random_normal_initializer(stddev=1.))
        npoints = tf.cast(tf.shape(sample_ph)[0], tf.int32)
        # First we need to make sure projection matrix is orthogonal
        v_norm = tf.nn.l2_normalize(v, 0)
        dotprod = tf.reduce_sum(tf.multiply(u, v_norm))
        u_ort = u - dotprod * v_norm
        u_norm = tf.nn.l2_normalize(u_ort, 0)
        Mproj = tf.concat([v_norm, u_norm], 1)
        sample_proj = tf.matmul(sample_ph, Mproj)
        a = tf.eye(npoints) - tf.ones([npoints, npoints]) / tf.cast(npoints, tf.float32)
        b = tf.matmul(sample_proj, tf.matmul(a, a), transpose_a=True)
        b = tf.matmul(b, sample_proj)
        # Sample covariance matrix
        covhat = b / (tf.cast(npoints, tf.float32) - 1)
        # covhat = tf.Print(covhat, [covhat], 'Cov:')
        with tf.variable_scope('leastGaussian2d'):
            gcov = opts['pot_pz_std'] * opts['pot_pz_std'] * tf.eye(2)
            # l2 distance between sample cov and the Gaussian cov
            projloss =  tf.reduce_sum(tf.square(covhat - gcov))
            # Also account for the first moment, i.e. expected value
            projloss += tf.reduce_sum(tf.square(tf.reduce_mean(sample_proj, 0)))
            # We are maximizing
            projloss = -projloss
            optim = tf.train.AdamOptimizer(0.001, 0.9)
            optim = optim.minimize(projloss, var_list=[v, u])

        self._proj_u = u_norm
        self._proj_v = v_norm
        self._proj_sample_ph = sample_ph
        self._proj_covhat = covhat
        self._proj_loss = projloss
        self._proj_optim = optim

    def least_gaussian_2d(self, opts, X):
        """
        Given a sample X of shape (n_points, n_z) find 2d plain
        such that projection looks least gaussian.
        """
        with self._session.as_default(), self._session.graph.as_default():
            sample_ph = self._proj_sample_ph
            optim = self._proj_optim
            loss = self._proj_loss
            u = self._proj_u
            v = self._proj_v
            covhat = self._proj_covhat
            proj_mat = tf.concat([v, u], 1).eval()
            dot_prod = -1
            best_of_runs = 10e5 # Any positive value would do
            updated = False
            for _start in xrange(3):
                # We will run 3 times from random inits
                loss_prev = 10e5 # Any positive value would do
                proj_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="leastGaussian2d")
                self._session.run(tf.variables_initializer(proj_vars))
                step = 0
                for _ in xrange(5000):
                    self._session.run(optim, feed_dict={sample_ph:X})
                    step += 1
                    if step % 10 == 0:
                        loss_cur = loss.eval(feed_dict={sample_ph: X})
                        rel_imp = abs(loss_cur - loss_prev) / abs(loss_prev)
                        if rel_imp < 1e-2:
                            break
                        loss_prev = loss_cur
                loss_final = loss.eval(feed_dict={sample_ph: X})
                if loss_final < best_of_runs:
                    updated = True
                    best_of_runs = loss_final
                    proj_mat = tf.concat([v, u], 1).eval()
                    dot_prod = tf.reduce_sum(tf.multiply(u, v)).eval()
        if not updated:
            logging.error('WARNING: possible bug in the worst 2d projection')
        return proj_mat, dot_prod

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to POT implementation.

        """
        data_shape = self._data.data_shape
        additional_losses = collections.OrderedDict()

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        enc_noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='enc_noise_ph')
        lr_decay_ph = tf.placeholder(tf.float32)
        is_training_ph = tf.placeholder(tf.bool, name='is_training_ph')
        keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

        # Operations
        if opts['pz_transform']:
            assert opts['z_test'] == 'gan', 'Pz transforms are currently allowed only for POT+GAN'
            noise = self.pz_sampler(opts, noise_ph)
        else:
            noise = noise_ph

        real_points = self._data_augmentation(
            opts, real_points_ph, is_training_ph)

        if opts['e_is_random']:
            # If encoder is random we map the training points
            # to the expectation of Q(Z|X) and then add the scaled
            # Gaussian noise corresponding to the learned sigmas
            enc_train_mean, enc_log_sigmas = self.encoder(
                opts, real_points,
                is_training=is_training_ph, keep_prob=keep_prob_ph)
            # enc_log_sigmas = tf.Print(enc_log_sigmas, [tf.reduce_max(enc_log_sigmas),
            #                                            tf.reduce_min(enc_log_sigmas),
            #                                            tf.reduce_mean(enc_log_sigmas)], 'Log sigmas:')
            # enc_log_sigmas = tf.Print(enc_log_sigmas, [tf.slice(enc_log_sigmas, [0,0], [1,-1])], 'Log sigmas:')
            # stds = tf.sqrt(tf.exp(enc_log_sigmas) + 1e-05)
            stds = tf.sqrt(tf.nn.relu(enc_log_sigmas) + 1e-05)
            # stds = tf.Print(stds, [stds[0], stds[1], stds[2], stds[3]], 'Stds: ')
            # stds = tf.Print(stds, [enc_train_mean[0], enc_train_mean[1], enc_train_mean[2]], 'Means: ')
            scaled_noise = tf.multiply(stds, enc_noise_ph)
            encoded_training = enc_train_mean + scaled_noise
        else:
            encoded_training = self.encoder(
                opts, real_points,
                is_training=is_training_ph, keep_prob=keep_prob_ph)
        reconstructed_training = self.generator(
            opts, encoded_training,
            is_training=is_training_ph, keep_prob=keep_prob_ph)
        reconstructed_training.set_shape(real_points.get_shape())

        if opts['recon_loss'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss_reconstr = tf.reduce_sum(
                tf.square(real_points - reconstructed_training), axis=1)
            # sqrt(x + delta) guarantees the direvative 1/(x + delta) is finite
            loss_reconstr = tf.reduce_mean(tf.sqrt(loss_reconstr + 1e-08))
        elif opts['recon_loss'] == 'l2f':
            # c(x,y) = ||x - y||_2
            loss_reconstr = tf.reduce_sum(
                tf.square(real_points - reconstructed_training), axis=[1, 2, 3])
            loss_reconstr = tf.reduce_mean(tf.sqrt(1e-08 + loss_reconstr)) * 0.2
        elif opts['recon_loss'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss_reconstr = tf.reduce_sum(
                tf.square(real_points - reconstructed_training), axis=[1, 2, 3])
            loss_reconstr = tf.reduce_mean(loss_reconstr) * 0.05
        elif opts['recon_loss'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss_reconstr = tf.reduce_mean(tf.reduce_sum(
                tf.abs(real_points - reconstructed_training), axis=[1, 2, 3])) * 0.02
        else:
            assert False

        # Pearson independence test of coordinates in Z space
        loss_z_corr = self.correlation_loss(opts, encoded_training)
        # Perform a Qz = Pz goodness of fit test based on Stein Discrepancy
        if opts['z_test'] == 'gan':
            # Pz = Qz test based on GAN in the Z space
            d_logits_Pz = self.discriminator(opts, noise)
            d_logits_Qz = self.discriminator(opts, encoded_training, reuse=True)
            d_loss_Pz = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logits_Pz, labels=tf.ones_like(d_logits_Pz)))
            d_loss_Qz = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logits_Qz, labels=tf.zeros_like(d_logits_Qz)))
            d_loss_Qz_trick = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logits_Qz, labels=tf.ones_like(d_logits_Qz)))
            d_loss = opts['pot_lambda'] * (d_loss_Pz + d_loss_Qz)
            if opts['pz_transform']:
                loss_match = d_loss_Qz_trick - d_loss_Pz
            else:
                loss_match = d_loss_Qz_trick
        elif opts['z_test'] == 'mmd':
            # Pz = Qz test based on MMD(Pz, Qz)
            loss_match = self.discriminator_mmd_test(opts, encoded_training, noise)
            d_loss = None
            d_logits_Pz = None
            d_logits_Qz = None
        elif opts['z_test'] == 'lks':
            # Pz = Qz test without adversarial training
            # based on Kernel Stein Discrepancy
            # Uncomment next line to check for the real Pz
            # loss_match = self.discriminator_test(opts, noise_ph)
            loss_match = self.discriminator_test(opts, encoded_training)
            d_loss = None
            d_logits_Pz = None
            d_logits_Qz = None
        else:
            # Pz = Qz test without adversarial training
            # (a) Check for multivariate Gaussianity
            #     by checking Gaussianity of all the 1d projections
            # (b) Run Pearson's test of coordinate independance
            loss_match = self.discriminator_test(opts, encoded_training)
            loss_match = loss_match + opts['z_test_corr_w'] * loss_z_corr
            d_loss = None
            d_logits_Pz = None
            d_logits_Qz = None
        g_mom_stats = self.moments_stats(opts, encoded_training)
        loss = opts['reconstr_w'] * loss_reconstr + opts['pot_lambda'] * loss_match

        # Optionally, add one more cost function based on the embeddings
        # add a discriminator in the X space, reusing the encoder or a new model.
        if opts['adv_c_loss'] == 'encoder':
            adv_c_loss, emb_c_loss = self._recon_loss_using_disc_encoder(
                opts, reconstructed_training, encoded_training, real_points, is_training_ph, keep_prob_ph)
            loss += opts['adv_c_loss_w'] * adv_c_loss + opts['emb_c_loss_w'] * emb_c_loss
            additional_losses['adv_c'], additional_losses['emb_c'] = adv_c_loss, emb_c_loss
        elif opts['adv_c_loss'] == 'conv':
            adv_c_loss, emb_c_loss = self._recon_loss_using_disc_conv(
                opts, reconstructed_training, real_points, is_training_ph, keep_prob_ph)
            additional_losses['adv_c'], additional_losses['emb_c'] = adv_c_loss, emb_c_loss
            loss += opts['adv_c_loss_w'] * adv_c_loss + opts['emb_c_loss_w'] * emb_c_loss
        elif opts['adv_c_loss'] == 'conv_eb':
            adv_c_loss, emb_c_loss = self._recon_loss_using_disc_conv_eb(
                opts, reconstructed_training, real_points, is_training_ph, keep_prob_ph)
            additional_losses['adv_c'], additional_losses['emb_c'] = adv_c_loss, emb_c_loss
            loss += opts['adv_c_loss_w'] * adv_c_loss + opts['emb_c_loss_w'] * emb_c_loss
        elif opts['adv_c_loss'] == 'vgg':
            emb_c_loss = self._recon_loss_using_vgg(
                opts, reconstructed_training, real_points, is_training_ph, keep_prob_ph)
            loss += opts['emb_c_loss_w'] * emb_c_loss
            additional_losses['emb_c'] = emb_c_loss
        elif opts['adv_c_loss'] == 'moments':
            emb_c_loss = self._recon_loss_using_moments(
                opts, reconstructed_training, real_points, is_training_ph, keep_prob_ph)
            loss += opts['emb_c_loss_w'] * emb_c_loss
            additional_losses['emb_c'] = emb_c_loss
        elif opts['adv_c_loss'] == 'vgg_moments':
            emb_c_loss = self._recon_loss_using_vgg_moments(
                opts, reconstructed_training, real_points, is_training_ph, keep_prob_ph)
            loss += opts['emb_c_loss_w'] * emb_c_loss
            additional_losses['emb_c'] = emb_c_loss
        else:
            assert opts['adv_c_loss'] == 'none'

        # Add ops to pretrain the Qz match mean and covariance of Pz
        loss_pretrain = None
        if opts['e_pretrain']:
            # Next two vectors are zdim-dimensional
            mean_pz = tf.reduce_mean(noise, axis=0, keep_dims=True)
            mean_qz = tf.reduce_mean(encoded_training, axis=0, keep_dims=True)
            mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
            cov_pz = tf.matmul(noise - mean_pz,
                               noise - mean_pz, transpose_a=True)
            cov_pz /= opts['e_pretrain_bsize'] - 1.
            cov_qz = tf.matmul(encoded_training - mean_qz,
                               encoded_training - mean_qz, transpose_a=True)
            cov_qz /= opts['e_pretrain_bsize'] - 1.
            cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
            loss_pretrain = mean_loss + cov_loss

        # Also add ops to find the least Gaussian 2d projection 
        # this is handy when visually inspection Qz = Pz
        self.add_least_gaussian2d_ops(opts)

        # Optimizer ops
        t_vars = tf.trainable_variables()
        # Updates for discriminator
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        # Updates for everything but adversary (encoder, decoder and possibly pz-transform)
        all_vars = [var for var in t_vars if 'DISCRIMINATOR/' not in var.name]
        # Updates for everything but adversary (encoder, decoder and possibly pz-transform)
        eg_vars = [var for var in t_vars if 'GENERATOR/' in var.name or 'ENCODER/' in var.name]
        # Encoder variables separately if we want to pretrain
        e_vars = [var for var in t_vars if 'ENCODER/' in var.name]

        logging.error('Param num in G and E: %d' % \
                np.sum([np.prod([int(d) for d in v.get_shape()]) for v in eg_vars]))
        for v in eg_vars:
            print v.name, [int(d) for d in v.get_shape()]

        if len(d_vars) > 0:
            d_optim = ops.optimizer(opts, net='d', decay=lr_decay_ph).minimize(loss=d_loss, var_list=d_vars)
        else:
            d_optim = None
        optim = ops.optimizer(opts, net='g', decay=lr_decay_ph).minimize(loss=loss, var_list=all_vars)
        pretrain_optim = None
        if opts['e_pretrain']:
            pretrain_optim = ops.optimizer(opts, net='g').minimize(loss=loss_pretrain, var_list=e_vars)


        generated_images = self.generator(
            opts, noise, is_training=is_training_ph,
            reuse=True, keep_prob=keep_prob_ph)

        self._real_points_ph = real_points_ph
        self._real_points = real_points
        self._noise_ph = noise_ph
        self._noise = noise
        self._enc_noise_ph = enc_noise_ph
        self._lr_decay_ph = lr_decay_ph
        self._is_training_ph = is_training_ph
        self._keep_prob_ph = keep_prob_ph
        self._optim = optim
        self._d_optim = d_optim
        self._pretrain_optim = pretrain_optim
        self._loss = loss
        self._loss_reconstruct = loss_reconstr
        self._loss_match = loss_match
        self._loss_z_corr = loss_z_corr
        self._loss_pretrain = loss_pretrain
        self._additional_losses = additional_losses
        self._g_mom_stats = g_mom_stats
        self._d_loss = d_loss
        self._generated = generated_images
        self._Qz = encoded_training
        self._reconstruct_x = reconstructed_training

        saver = tf.train.Saver(max_to_keep=10)
        tf.add_to_collection('real_points_ph', self._real_points_ph)
        tf.add_to_collection('noise_ph', self._noise_ph)
        tf.add_to_collection('enc_noise_ph', self._enc_noise_ph)
        if opts['pz_transform']:
            tf.add_to_collection('noise', self._noise)
        tf.add_to_collection('is_training_ph', self._is_training_ph)
        tf.add_to_collection('keep_prob_ph', self._keep_prob_ph)
        tf.add_to_collection('encoder', self._Qz)
        tf.add_to_collection('decoder', self._generated)
        if d_logits_Pz is not None:
            tf.add_to_collection('disc_logits_Pz', d_logits_Pz)
        if d_logits_Qz is not None:
            tf.add_to_collection('disc_logits_Qz', d_logits_Qz)

        self._saver = saver

        logging.error("Building Graph Done.")

    def pretrain(self, opts):
        steps_max = 200
        batch_size = opts['e_pretrain_bsize']
        for step in xrange(steps_max):
            train_size = self._data.num_points
            data_ids = np.random.choice(train_size, min(train_size, batch_size),
                                        replace=False)
            batch_images = self._data.data[data_ids].astype(np.float)
            batch_noise = opts['pot_pz_std'] *\
                utils.generate_noise(opts, batch_size)
            # Noise for the random encoder (if present)
            batch_enc_noise = utils.generate_noise(opts, batch_size)

            # Update encoder
            [_, loss_pretrain] = self._session.run(
                [self._pretrain_optim,
                 self._loss_pretrain],
                feed_dict={self._real_points_ph: batch_images,
                           self._noise_ph: batch_noise,
                           self._enc_noise_ph: batch_enc_noise,
                           self._is_training_ph: True,
                           self._keep_prob_ph: opts['dropout_keep_prob']})

            if opts['verbose'] == 2:
                logging.error('Step %d/%d, loss=%f' % (step, steps_max, loss_pretrain))

            if loss_pretrain < 0.1:
                break

    def _train_internal(self, opts):
        """Train a POT model.

        """
        logging.error(opts)

        batches_num = self._data.num_points / opts['batch_size']
        train_size = self._data.num_points
        num_plot = 320
        sample_prev = np.zeros([num_plot] + list(self._data.data_shape))
        l2s = []
        losses = []
        losses_rec = []
        losses_match = []
        wait = 0

        start_time = time.time()
        counter = 0
        decay = 1.
        logging.error('Training POT')

        # Optionally we first pretrain the Qz to match mean and
        # covariance of Pz
        if opts['e_pretrain']:
            logging.error('Pretraining the encoder')
            self.pretrain(opts)
            logging.error('Pretraining the encoder done')

        for _epoch in xrange(opts["gan_epoch_num"]):

            if opts['decay_schedule'] == "manual":
                if _epoch == 30:
                    decay = decay / 2.
                if _epoch == 50:
                    decay = decay / 5.
                if _epoch == 100:
                    decay = decay / 10.
            elif opts['decay_schedule'] != "plateau":
                assert type(1.0 * opts['decay_schedule']) == float
                decay = 1.0 * 10**(-_epoch / float(opts['decay_schedule']))

            if _epoch > 0 and _epoch % opts['save_every_epoch'] == 0:
                os.path.join(opts['work_dir'], opts['ckpt_dir'])
                self._saver.save(self._session,
                                 os.path.join(opts['work_dir'],
                                              opts['ckpt_dir'],
                                              'trained-pot'),
                                 global_step=counter)

            for _idx in xrange(batches_num):
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=self._data_weights)
                batch_images = self._data.data[data_ids].astype(np.float)
                # Noise for the Pz=Qz GAN
                batch_noise = opts['pot_pz_std'] *\
                    utils.generate_noise(opts, opts['batch_size'])
                # Noise for the random encoder (if present)
                batch_enc_noise = utils.generate_noise(opts, opts['batch_size'])

                # Update generator (decoder) and encoder
                [_, loss, loss_rec, loss_match] = self._session.run(
                    [self._optim,
                     self._loss,
                     self._loss_reconstruct,
                     self._loss_match],
                    feed_dict={self._real_points_ph: batch_images,
                               self._noise_ph: batch_noise,
                               self._enc_noise_ph: batch_enc_noise,
                               self._lr_decay_ph: decay,
                               self._is_training_ph: True,
                               self._keep_prob_ph: opts['dropout_keep_prob']})

                if opts['decay_schedule'] == "plateau":
                    # First 30 epochs do nothing
                    if _epoch >= 30:
                        # If no significant progress was made in last 10 epochs
                        # then decrease the learning rate.
                        if loss < min(losses[-20 * batches_num:]):
                            wait = 0
                        else:
                            wait += 1
                        if wait > 10 * batches_num:
                            decay = max(decay  / 1.4, 1e-6)
                            logging.error('Reduction in learning rate: %f' % decay)
                            wait = 0
                losses.append(loss)
                losses_rec.append(loss_rec)
                losses_match.append(loss_match)
                if opts['verbose'] >= 2:
                    # logging.error('loss after %d steps : %f' % (counter, losses[-1]))
                    logging.error('loss match  after %d steps : %f' % (counter, losses_match[-1]))

                # Update discriminator in Z space (if any).
                if self._d_optim is not None:
                    for _st in range(opts['d_steps']):
                        if opts['d_new_minibatch']:
                            d_data_ids = np.random.choice(
                                train_size, opts['batch_size'],
                                replace=False, p=self._data_weights)
                            d_batch_images = self._data.data[data_ids].astype(np.float)
                            d_batch_enc_noise = utils.generate_noise(opts, opts['batch_size'])
                        else:
                            d_batch_images = batch_images
                            d_batch_enc_noise = batch_enc_noise
                        _ = self._session.run(
                            [self._d_optim, self._d_loss],
                            feed_dict={self._real_points_ph: d_batch_images,
                                       self._noise_ph: batch_noise,
                                       self._enc_noise_ph: d_batch_enc_noise,
                                       self._lr_decay_ph: decay,
                                       self._is_training_ph: True,
                                       self._keep_prob_ph: opts['dropout_keep_prob']})
                counter += 1
                now = time.time()

                rec_test = None
                if opts['verbose'] and counter % 500 == 0:
                    # Printing (training and test) loss values
                    test = self._data.test_data[:200]
                    [loss_rec_test, rec_test, g_mom_stats, loss_z_corr, additional_losses] = self._session.run(
                        [self._loss_reconstruct, self._reconstruct_x, self._g_mom_stats, self._loss_z_corr,
                         self._additional_losses],
                        feed_dict={self._real_points_ph: test,
                                   self._enc_noise_ph: utils.generate_noise(opts, len(test)),
                                   self._is_training_ph: False,
                                   self._noise_ph: batch_noise,
                                   self._keep_prob_ph: 1e5})
                    debug_str = 'Epoch: %d/%d, batch:%d/%d, batch/sec:%.2f' % (
                        _epoch+1, opts['gan_epoch_num'], _idx+1,
                        batches_num, float(counter) / (now - start_time))
                    debug_str += '  [L=%.5f, Recon=%.5f, GanL=%.5f, Recon_test=%.5f' % (
                        loss, loss_rec, loss_match, loss_rec_test)
                    debug_str += ',' + ', '.join(
                        ['%s=%.2g' % (k, v) for (k, v) in additional_losses.items()])
                    logging.error(debug_str)
                    if opts['verbose'] >= 2:
                        logging.error(g_mom_stats)
                        logging.error(loss_z_corr)
                    if counter % opts['plot_every'] == 0:
                        # plotting the test images.
                        metrics = Metrics()
                        merged = np.vstack([rec_test[:8 * 10], test[:8 * 10]])
                        r_ptr = 0
                        w_ptr = 0
                        for _ in range(8 * 10):
                            merged[w_ptr] = test[r_ptr]
                            merged[w_ptr + 1] = rec_test[r_ptr]
                            r_ptr += 1
                            w_ptr += 2
                        metrics.make_plots(
                            opts,
                            counter,
                            None,
                            merged,
                            prefix='test_reconstr_e%04d_mb%05d_' % (_epoch, _idx))

                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    # Plotting intermediate results
                    metrics = Metrics()
                    # --Random samples from the model
                    points_to_plot, sample_pz = self._session.run(
                        [self._generated, self._noise],
                        feed_dict={
                            self._noise_ph: self._noise_for_plots[0:num_plot],
                            self._is_training_ph: False,
                            self._keep_prob_ph: 1e5})
                    Qz_num = 320
                    sample_Qz = self._session.run(
                        self._Qz,
                        feed_dict={
                            self._real_points_ph: self._data.data[:Qz_num],
                            self._enc_noise_ph: utils.generate_noise(opts, Qz_num),
                            self._is_training_ph: False,
                            self._keep_prob_ph: 1e5})
                    # Searching least Gaussian 2d projection
                    proj_mat, check = self.least_gaussian_2d(opts, sample_Qz)
                    # Projecting samples from Qz and Pz on this 2d plain
                    metrics.Qz = np.dot(sample_Qz, proj_mat)
                    # metrics.Pz = np.dot(self._noise_for_plots, proj_mat)
                    metrics.Pz = np.dot(sample_pz, proj_mat)
                    if self._data.labels != None:
                        metrics.Qz_labels = self._data.labels[:Qz_num]
                    else:
                        metrics.Qz_labels = None
                    metrics.l2s = losses[:]
                    metrics.losses_match = [opts['pot_lambda'] * el for el in losses_match]
                    metrics.losses_rec = [opts['reconstr_w'] * el for el in losses_rec]
                    to_plot = [points_to_plot, 0 * batch_images[:16], batch_images]
                    if rec_test is not None:
                        to_plot += [0 * batch_images[:16], rec_test[:64]]
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        np.vstack(to_plot),
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx) if rec_test is None \
                                else 'sample_with_test_e%04d_mb%05d_' % (_epoch, _idx))


                    # --Reconstructions for the train and test points
                    num_real_p = 8 * 10
                    reconstructed, real_p = self._session.run(
                        [self._reconstruct_x, self._real_points],
                        feed_dict={
                            self._real_points_ph: self._data.data[:num_real_p],
                            self._enc_noise_ph: utils.generate_noise(opts, num_real_p),
                            self._is_training_ph: True,
                            self._keep_prob_ph: 1e5})
                    points = real_p
                    merged = np.vstack([reconstructed, points])
                    r_ptr = 0
                    w_ptr = 0
                    for _ in range(8 * 10):
                        merged[w_ptr] = points[r_ptr]
                        merged[w_ptr + 1] = reconstructed[r_ptr]
                        r_ptr += 1
                        w_ptr += 2
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        merged,
                        prefix='reconstr_e%04d_mb%05d_' % (_epoch, _idx))
                    sample_prev = points_to_plot[:]
        if _epoch > 0:
            os.path.join(opts['work_dir'], opts['ckpt_dir'])
            self._saver.save(self._session,
                             os.path.join(opts['work_dir'],
                                          opts['ckpt_dir'],
                                          'trained-pot-final'),
                             global_step=counter)

    def _sample_internal(self, opts, num):
        """Sample from the trained GAN model.

        """
        # noise = opts['pot_pz_std'] * utils.generate_noise(opts, num)
        # sample = self._run_batch(
        #     opts, self._generated, self._noise_ph, noise, self._is_training_ph, False)
        sample = None
        return sample
