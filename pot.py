# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class implements POT training.

"""
import collections
import logging
import os
import tensorflow as tf
import utils
from utils import ProgressBar
from utils import TQDM
import numpy as np
import ops
from metrics import Metrics
slim = tf.contrib.slim

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
        self._noise_for_plots = opts['pot_pz_std'] * utils.generate_noise(opts, 500)
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
                h0 = ops.linear(opts, noise, num_units, 'h0_lin')
                h0 = tf.nn.relu(h0)
                h1 = ops.linear(opts, h0, num_units, 'h1_lin')
                h1 = tf.nn.relu(h1)
                h2 = ops.linear(opts, h1, num_units, 'h2_lin')
                h2 = tf.nn.relu(h2)
                h3 = ops.linear(opts, h2, np.prod(output_shape), 'h3_lin')
                h3 = tf.reshape(h3, [-1] + list(output_shape))
                if opts['input_normalize_sym']:
                    return tf.nn.tanh(h3)
                else:
                    return tf.nn.sigmoid(h3)
            elif opts['g_arch'] in ['dcgan', 'dcgan_mod']:
                return self.dcgan_like_arch(opts, noise, is_training, reuse, keep_prob)
            elif opts['g_arch'] == 'conv_up_res':
                return self.conv_up_res(opts, noise, is_training, reuse, keep_prob)
            elif opts['g_arch'] == 'ali':
                return self.ali_deconv(opts, noise, is_training, reuse, keep_prob)
            else:
                raise ValueError('%s unknown' % opts['g_arch'])

    def discriminator(self, opts, input_, prefix='DISCRIMINATOR', reuse=False):
        """Discriminator for the GAN objective

        """

        num_units = opts['d_num_filters']
        num_layers = opts['d_num_layers']
        # No convolutions as GAN happens in the latent space
        with tf.variable_scope(prefix, reuse=reuse):
            hi = ops.linear(opts, input_, num_units, scope='h0_lin')
            for i in range(num_layers-1):
                hi = tf.nn.relu(hi)
                hi = ops.linear(opts, hi, num_units, scope='h%d_lin' % (i+1))

        return hi

    def get_batch_size(self, opts, input_):
        return tf.cast(tf.shape(input_)[0], tf.float32)# opts['batch_size']

    def moments_stats(self, opts, input_):
        input_ = input_ / opts['pot_pz_std']
        p1 = tf.reduce_mean(input_, 0)
        center_inp = input_ - p1
        p2 = tf.sqrt(1e-5 + tf.reduce_mean(tf.square(center_inp), 0))
        normed_inp = center_inp / p2
        p3 = tf.pow(1e-5 + tf.abs(tf.reduce_mean(tf.pow(normed_inp, 3), 0)), 1.0 / 3.0)
        # Because 3 is the right Kurtosis for N(0, 1)
        p4 = tf.pow(1e-5 + tf.reduce_mean(tf.pow(normed_inp, 4), 0) / 3.0, 1.0 / 4.0)
        def zero_t(v):
            return tf.sqrt(1e-5 + tf.reduce_mean(tf.square(v)))
        def one_t(v):
            return tf.sqrt(1e-5 + tf.reduce_mean(tf.maximum(tf.square(v), 1.0 / (1e-5 + tf.square(v)))))
        return tf.stack([zero_t(p1), one_t(p2), zero_t(p3), one_t(p4)])
#         return tf.stack([mse(p1), mse(p2), mse(p3), mse(p4)])

    def discriminator_test(self, opts, input_):
        """Deterministic discriminator using simple tests."""
        if opts['z_test'] == 'cramer':
            test_v = self.discriminator_cramer_test(opts, input_)
        elif opts['z_test'] == 'anderson':
            test_v = self.discriminator_anderson_test(opts, input_)
        elif opts['z_test'] == 'moments':
            test_v = tf.reduce_mean(self.moments_stats(opts, input_)) / 10.0
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
        # top_k can only sort on the last dimension and we want to sort the
        # first one (batch_size).
        batch_size = self.get_batch_size(opts, input_)
        transposed = tf.transpose(input_, perm=[1, 0])
        values, indices = tf.nn.top_k(transposed, k=tf.cast(batch_size, tf.int32))
        values = tf.reverse(values, [1])
        #values = tf.Print(values, [values], "sorted values")
        normal_dist = tf.contrib.distributions.Normal(0., float(opts['pot_pz_std']))
        #
        normal_cdf = normal_dist.cdf(values)
        #normal_cdf = tf.Print(normal_cdf, [normal_cdf], "normal_cdf")
        ln_normal_cdf = tf.log(normal_cdf)
        ln_one_normal_cdf = tf.log(1.0 - normal_cdf)
        w1 = 2 * tf.range(1, batch_size+1, 1, dtype="float") - 1
        w2 = 2 * tf.range(batch_size-1, -1, -1, dtype="float") + 1
        stat = -batch_size - tf.reduce_sum(w1 * ln_normal_cdf + w2 * ln_one_normal_cdf, 1) / batch_size
        stat = tf.reduce_mean(stat)
        #stat = tf.Print(stat, [stat], "stat")
        return stat

    def correlation_loss(self, opts, input_):
        batch_size = self.get_batch_size(opts, input_)
        dim = int(input_.get_shape()[1])
        transposed = tf.transpose(input_, perm=[1, 0])
        print("transposed shape", transposed.get_shape())
        mean = tf.reshape(tf.reduce_mean(transposed, axis=1), [-1, 1])
        centered_transposed = transposed - mean
        cov = tf.matmul(centered_transposed, tf.transpose(centered_transposed)) / batch_size
        #cov = tf.Print(cov, [cov], "cov")
        sigmas = tf.sqrt(tf.diag_part(cov) + 1e-5)
        #sigmas = tf.Print(sigmas, [sigmas], "sigmas")
        sigmas = tf.reshape(sigmas, [1, -1])
        sigmas = tf.matmul(tf.transpose(sigmas), sigmas)
        #sigmas = tf.Print(sigmas, [sigmas], "sigmas")
        corr = cov / sigmas
        triangle = tf.matrix_set_diag(tf.matrix_band_part(corr, 0, -1), tf.zeros(dim))
        #triangle = tf.Print(triangle, [triangle], "triangle")
        loss = tf.reduce_sum(tf.square(triangle)) / ((dim * dim - dim) / 2.0)
        #loss = tf.Print(loss, [loss], "Correlation loss")
        return loss


    def encoder(self, opts, input_, is_training=False, reuse=False, keep_prob=1.):

        num_units = opts['g_num_filters']
        with tf.variable_scope("ENCODER", reuse=reuse):
            if not opts['convolutions']:
                h0 = ops.linear(opts, input_, 1024, 'h0_lin')
                h0 = tf.nn.relu(h0)
                h1 = ops.linear(opts, h0, 512, 'h1_lin')
                h1 = tf.nn.relu(h1)
                h2 = ops.linear(opts, h1, 512, 'h2_lin')
                h2 = tf.nn.relu(h2)
                return ops.linear(opts, h2, opts['latent_space_dim'], 'h3_lin')
            elif opts['e_arch'] == 'dcgan':
                return self.dcgan_encoder(opts, input_, is_training, reuse, keep_prob)
            elif opts['e_arch'] == 'ali':
                return self.ali_encoder(opts, input_, is_training, reuse, keep_prob)
            else:
                raise ValueError('%s Unknown' % opts['e_arch'])

    def dcgan_encoder(self, opts, input_, is_training=False, reuse=False, keep_prob=1.):
        num_units = opts['g_num_filters']
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

        return ops.linear(opts, layer_x, opts['latent_space_dim'], scope='hlast_lin')

    def ali_encoder(self, opts, input_, is_training=False, reuse=False, keep_prob=1.):
        num_units = opts['g_num_filters']
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
            print((height, width))
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

        return ops.linear(opts, layer_x, opts['latent_space_dim'], scope='hlast_lin')

    def _data_augmentation(self, opts, real_points, is_training):
        if not opts['data_augm']:
            return real_points

        height = int(real_points.get_shape()[1])
        width = int(real_points.get_shape()[2])
        depth = int(real_points.get_shape()[3])
        print("real_points shape", real_points.get_shape())
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

    def _recon_loss_using_disc_encoder(self, opts, reconstructed_training, encoded_training, real_points, is_training_ph, keep_prob_ph):
        """Build an additional loss using the encoder as discriminator."""
        reconstructed_reencoded_sg = self.encoder(
            opts, tf.stop_gradient(reconstructed_training), is_training=is_training_ph, keep_prob=keep_prob_ph, reuse=True)
        reconstructed_reencoded = self.encoder(
            opts, reconstructed_training, is_training=is_training_ph, keep_prob=keep_prob_ph, reuse=True)
        # Below line enforces the forward to be reconstructed_reencoded and backwards to NOT change the encoder....
        crazy_hack = reconstructed_reencoded-reconstructed_reencoded_sg+tf.stop_gradient(reconstructed_reencoded_sg)
        encoded_training_sg = self.encoder(
            opts, tf.stop_gradient(real_points),
            is_training=is_training_ph, keep_prob=keep_prob_ph, reuse=True)

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
        def _architecture(layer_x, reuse=None):
            with tf.variable_scope('DISC_X_LOSS', reuse=reuse):
                num_units = 128
                num_layers = 2
                for i in xrange(num_layers):
                    scale = 2**(num_layers-i-1)
                    layer_x = ops.conv2d(opts, layer_x, num_units / scale, scope='h%d_conv' % i)
                    if opts['batch_norm']:
                        layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d' % i)
                    layer_x = tf.nn.relu(layer_x)
                size = int(layer_x.get_shape()[1])
                last = ops.conv2d(
                    opts, layer_x, 1, d_h=1, d_w=1, scope="last_lin",
                    conv_filters_dim=opts['adv_c_patches_size'])
                return layer_x, tf.reshape(last, [-1, size * size])


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
        # Note: the reduce on axis 1 does not really make sense as those tensors
        # are of shape [batch_size, height, width, num_filters]. But we keep it
        # that way to be similar to the L2 reconstruction cost.
        emb_c = tf.reduce_sum(tf.square(crazy_hack - tf.stop_gradient(real_p_embed)), 1)
        emb_c_loss = tf.reduce_mean(tf.sqrt(emb_c + 1e-5))
        # Normalize the loss, so that it does not depend on how good the
        # discriminator is.
        emb_c_loss = emb_c_loss / tf.stop_gradient(emb_c_loss)
        return adv_c_loss, emb_c_loss

    def _recon_loss_using_vgg(self, opts, reconstructed_training, real_points, is_training, keep_prob):
        """Build an additional loss using a pretrained VGG in X space."""
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

        def _architecture(_inputs, reuse=None):
            _, end_points = vgg_16(_inputs, is_training=is_training, dropout_keep_prob=keep_prob, reuse=reuse)
            layer_name = opts['vgg_layer']
            if layer_name == 'concat':
                outputs = []
                for ln in ['pool1', 'pool2', 'pool3']:
                    output = end_points[ln]
                    output = tf.reshape(output, [-1, np.prod([int(d) for d in output.get_shape()[1:]])])
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
                    output = tf.reshape(output, [-1, np.prod([int(d) for d in output.get_shape()[1:]])])
                    # We sqrt the weight here because we use L2 after.
                    outputs.append(np.sqrt(float(weights[lnum])) * output)
                output = tf.concat(outputs, 1)
            else:
                output = end_points[layer_name]
                output = tf.reshape(output, [-1, np.prod([int(d) for d in output.get_shape()[1:]])])
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
            _inputs_sq = tf.square(_inputs)
            height = int(_inputs.get_shape()[1])
            width = int(_inputs.get_shape()[2])
            channels = int(_inputs.get_shape()[3])
            def ConvFlatten(x, kernel_size):
#                 w_sum = tf.ones([kernel_size, kernel_size, channels, 1]) / (kernel_size * kernel_size * channels)
                w_sum = tf.eye(num_rows=channels, num_columns=channels, batch_shape=[kernel_size * kernel_size])
                w_sum = tf.reshape(w_sum, [kernel_size, kernel_size, channels, channels])
                w_sum = w_sum / (kernel_size * kernel_size)
                sum_ = tf.nn.conv2d(x, w_sum, strides=[1, 1, 1, 1], padding='VALID')
                size = np.prod([int(d) for d in sum_.get_shape()[1:]])
                assert size == (height - kernel_size + 1) * (width - kernel_size + 1) * channels, size
                return tf.reshape(sum_, [-1, size])
            outputs = []
            for size in [5]:#, 4, 5]:  # TODO: tune
                mean = ConvFlatten(_inputs, size)
                square = ConvFlatten(_inputs_sq, size)
                var = square - tf.square(mean)
#                 outputs += [mean, tf.sqrt(1e-5 + var)]
                outputs += [var]
            return tf.concat(outputs, 1)

        reconstructed_embed = _architecture(reconstructed_training)
        real_p_embed = _architecture(real_points)

        emb_c = tf.reduce_mean(tf.square(reconstructed_embed - tf.stop_gradient(real_p_embed)), 1)
#         emb_c = tf.Print(emb_c, [emb_c], "emb_c")
        emb_c_loss = tf.reduce_mean(emb_c)
        return emb_c_loss * 100.0 * 100.0 # TODO: constant

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
        lr_decay_ph = tf.placeholder(tf.float32)
        is_training_ph = tf.placeholder(tf.bool, name='is_training_ph')
        keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

        # Operations
        real_points = self._data_augmentation(opts, real_points_ph, is_training_ph)

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

        loss_z_corr = self.correlation_loss(opts, encoded_training)
        if opts['z_test'] == 'gan':
            d_logits_Pz = self.discriminator(opts, noise_ph)
            d_logits_Qz = self.discriminator(opts, encoded_training, reuse=True)
            d_loss_Pz = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logits_Pz, labels=tf.ones_like(d_logits_Pz)))
            d_loss_Qz = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logits_Qz, labels=tf.zeros_like(d_logits_Qz)))
            d_loss = opts['pot_lambda'] * (d_loss_Pz + d_loss_Qz)

            loss_gan = -d_loss_Qz
        else:
            d_loss = None
            loss_gan = self.discriminator_test(opts, encoded_training)
            loss_gan = loss_gan + opts['z_test_corr_w'] * loss_z_corr
            d_logits_Pz = None
            d_logits_Qz = None
        g_mom_stats = self.moments_stats(opts, encoded_training)
        loss = opts['reconstr_w'] * loss_reconstr + opts['pot_lambda'] * loss_gan

        # Optionally, add a discriminator in the X space, reusing the encoder.
        if opts['adv_c_loss'] == 'encoder':
            adv_c_loss, emb_c_loss = self._recon_loss_using_disc_encoder(
                opts, reconstructed_training, encoded_training, real_points, is_training_ph, keep_prob_ph)
            loss += opts['adv_c_loss_w'] * adv_c_loss + opts['emb_c_loss_w'] * emb_c_loss
            additional_losses['adv_c'], additional_losses['emb_c'] = adv_c_loss, emb_c_loss
        elif opts['adv_c_loss'] == 'conv':
            adv_c_loss, emb_c_loss = self._recon_loss_using_disc_conv(
                opts, reconstructed_training, real_points, is_training_ph, keep_prob_ph)
            loss += opts['adv_c_loss_w'] * adv_c_loss + opts['emb_c_loss_w'] * emb_c_loss
            additional_losses['adv_c'], additional_losses['emb_c'] = adv_c_loss, emb_c_loss
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
        else:
            assert opts['adv_c_loss'] == 'none'

        t_vars = tf.trainable_variables()
        # Updates for discriminator
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        # Updates for encoder and generator
        eg_vars = [var for var in t_vars if 'DISCRIMINATOR/' not in var.name]

        if len(d_vars) > 0:
            d_optim = ops.optimizer(opts, net='d', decay=lr_decay_ph).minimize(loss=d_loss, var_list=d_vars)
        else:
            d_optim = None
        optim = ops.optimizer(opts, net='g', decay=lr_decay_ph).minimize(loss=loss, var_list=eg_vars)

        generated_images = self.generator(
            opts, noise_ph, is_training=is_training_ph,
            reuse=True, keep_prob=keep_prob_ph)

        self._real_points_ph = real_points_ph
        self._real_points = real_points
        self._noise_ph = noise_ph
        self._lr_decay_ph = lr_decay_ph
        self._is_training_ph = is_training_ph
        self._keep_prob_ph = keep_prob_ph
        self._optim = optim
        self._d_optim = d_optim
        self._loss = loss
        self._loss_reconstruct = loss_reconstr
        self._loss_gan = loss_gan
        self._loss_z_corr = loss_z_corr
        self._additional_losses = additional_losses
        self._g_mom_stats = g_mom_stats
        self._d_loss = d_loss
        self._generated = generated_images
        self._Qz = encoded_training
        self._reconstruct_x = reconstructed_training

        saver = tf.train.Saver()
        tf.add_to_collection('real_points_ph', self._real_points_ph)
        tf.add_to_collection('noise_ph', self._noise_ph)
        tf.add_to_collection('is_training_ph', self._is_training_ph)
        tf.add_to_collection('keep_prob_ph', self._is_training_ph)
        tf.add_to_collection('encoder', self._Qz)
        tf.add_to_collection('decoder', self._generated)
        if d_logits_Pz is not None:
            tf.add_to_collection('disc_logits_Pz', d_logits_Pz)
        if d_logits_Qz is not None:
            tf.add_to_collection('disc_logits_Qz', d_logits_Qz)

        self._saver = saver

        logging.error("Building Graph Done.")


    def _train_internal(self, opts):
        """Train a POT model.

        """

        batches_num = self._data.num_points / opts['batch_size']
        train_size = self._data.num_points
        num_plot = 320
        sample_prev = np.zeros([num_plot] + list(self._data.data_shape))
        l2s = []
        losses = []

        counter = 0
        decay = 1.
        logging.error('Training POT')

        for _epoch in xrange(opts["gan_epoch_num"]):

            if opts['decay_schedule'] == "manual":
                if _epoch == 30:
                    decay = decay / 5.
                if _epoch == 50:
                    decay = decay / 10.
                if _epoch == 100:
                    decay = decay / 100.
            else:
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
                # logging.error('Step %d of %d' % (_idx, batches_num ) )
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=self._data_weights)
                batch_images = self._data.data[data_ids].astype(np.float)
                batch_noise = opts['pot_pz_std'] * utils.generate_noise(opts, opts['batch_size'])


                # Update generator (decoder) and encoder
                [_, loss, loss_rec, loss_gan] = self._session.run(
                    [self._optim,
                     self._loss,
                     self._loss_reconstruct,
                     self._loss_gan],
                    feed_dict={self._real_points_ph: batch_images,
                               self._noise_ph: batch_noise,
                               self._lr_decay_ph: decay,
                               self._is_training_ph: True,
                               self._keep_prob_ph: opts['dropout_keep_prob']})
                losses.append(loss)

                # Update discriminator in Z space (if any).
                if self._d_optim is not None:
                    for _st in range(opts['d_steps']):
                        if opts['d_new_minibatch']:
                            d_data_ids = np.random.choice(
                                train_size, opts['batch_size'],
                                replace=False, p=self._data_weights)
                            d_batch_images = self._data.data[data_ids].astype(np.float)
                        else:
                            d_batch_images = batch_images
                        _ = self._session.run(
                            [self._d_optim, self._d_loss],
                            feed_dict={self._real_points_ph: d_batch_images,
                                       self._noise_ph: batch_noise,
                                       self._lr_decay_ph: decay,
                                       self._is_training_ph: True,
                                       self._keep_prob_ph: opts['dropout_keep_prob']})
                counter += 1

                rec_test = None
                if opts['verbose'] and counter % 100 == 0:
                    # Printing (training and test) loss values
                    test = self._data.test_data
                    [loss_rec_test, rec_test, g_mom_stats, loss_z_corr, additional_losses] = self._session.run(
                        [self._loss_reconstruct,
                         self._reconstruct_x, self._g_mom_stats, self._loss_z_corr, self._additional_losses],
                        feed_dict={self._real_points_ph: test,
                                   self._is_training_ph: False,
                                   self._keep_prob_ph: 1e5})
                    debug_str = 'Epoch: %d/%d, batch:%d/%d' % (
                        _epoch+1, opts['gan_epoch_num'], _idx+1, batches_num)
                    debug_str += '  [L=%.2g, Recon=%.2g, GanL=%.2g, Recon_test=%.2g' % (
                        loss, loss_rec, loss_gan, loss_rec_test)
                    debug_str += ',' + ', '.join(['%s=%.2g' % (k, v) for (k, v) in additional_losses.items()])
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
                    points_to_plot = self._session.run(
                        self._generated,
                        feed_dict={
                            self._noise_ph: self._noise_for_plots[0:num_plot],
                            self._is_training_ph: False,
                            self._keep_prob_ph: 1e5})
                    metrics.Qz = self._session.run(
                        self._Qz,
                        feed_dict={
                            self._real_points_ph: self._data.data[:1000],
                            self._is_training_ph: False,
                            self._keep_prob_ph: 1e5})
                    metrics.Qz_labels = self._data.labels[:1000]
                    metrics.Pz = batch_noise
                    # l2s.append(np.sum((points_to_plot - sample_prev)**2))
                    # metrics.l2s = l2s[:]
                    metrics.l2s = losses[:]
                    to_plot = [points_to_plot, 0 * batch_images[:16], batch_images]
                    if rec_test is not None:
                        to_plot += [0 * batch_images[:16], rec_test[:64]]
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        np.vstack(to_plot),
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))

                    # --Reconstructions for the train and test points
                    reconstructed, real_p = self._session.run(
                        [self._reconstruct_x, self._real_points],
                        feed_dict={
                            self._real_points_ph: self._data.data[:8 * 10],
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

    def _sample_internal(self, opts, num):
        """Sample from the trained GAN model.

        """
        # noise = opts['pot_pz_std'] * utils.generate_noise(opts, num)
        # sample = self._run_batch(
        #     opts, self._generated, self._noise_ph, noise, self._is_training_ph, False)
        sample = None
        return sample
