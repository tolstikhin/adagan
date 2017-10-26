# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class implements VAE training.

"""

import os
import logging
import tensorflow as tf
import utils
from utils import ProgressBar
from utils import TQDM
import numpy as np
import ops
from metrics import Metrics

class Vae(object):
    """A base class for running individual VAEs.

    """
    def __init__(self, opts, data, weights):

        # Create a new session with session.graph = default graph
        self._session = tf.Session()
        self._trained = False
        self._data = data
        self._data_weights = np.copy(weights)
        # Latent noise sampled ones to apply decoder while training
        self._noise_for_plots = utils.generate_noise(opts, 500)
        # Placeholders
        self._real_points_ph = None
        self._noise_ph = None

        # Main operations
        # FIX
        self._loss = None
        self._loss_reconstruct = None
        self._loss_kl = None
        self._generated = None
        self._reconstruct_x = None

        # Optimizers
        self.optim = None

        with self._session.as_default(), self._session.graph.as_default():
            logging.error('Building the graph...')
            self._build_model_internal(opts)

        # Make sure AdamOptimizer, if used in the Graph, is defined before
        # calling global_variables_initializer().
        init = tf.global_variables_initializer()
        self._session.run(init)

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
        """Train a VAE model.

        """
        with self._session.as_default(), self._session.graph.as_default():
            self._train_internal(opts)
            self._trained = True

    def sample(self, opts, num=100):
        """Sample points from the trained VAE model.

        """
        assert self._trained, 'Can not sample from the un-trained VAE'
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
        assert False, 'VAE base class has no build_model method defined.'

    def _train_internal(self, opts):
        assert False, 'VAE base class has no train method defined.'

    def _sample_internal(self, opts, num):
        assert False, 'VAE base class has no sample method defined.'

    def _train_mixture_discriminator_internal(self, opts, fake_images):
        assert False, 'VAE base class has no mixture discriminator method defined.'


class ImageVae(Vae):
    """A simple VAE implementation, suitable for pictures.

    """

    def __init__(self, opts, data, weights):

        # One more placeholder for batch norm
        self._is_training_ph = None

        Vae.__init__(self, opts, data, weights)

    def generator(self, opts, noise, is_training, reuse=False, return_logits=False):
        """Generator function, suitable for simple picture experiments.

        Args:
            noise: [num_points, dim] array, where dim is dimensionality of the
                latent noise space.
            is_training: bool, defines whether to use batch_norm in the train
                or test mode.
            return_logits: bool, if true returns the "logits" instead of being
                normalized (by tanh or sigmoid depending on "input_normalize_sym".
        Returns:
            [num_points, dim1, dim2, dim3] array, where the first coordinate
            indexes the points, which all are of the shape (dim1, dim2, dim3).
        """

        output_shape = self._data.data_shape # (dim1, dim2, dim3)
        # Computing the number of noise vectors on-the-go
        dim1 = tf.shape(noise)[0]
        num_filters = opts['g_num_filters']
        num_layers = opts['g_num_layers']

        with tf.variable_scope("GENERATOR", reuse=reuse):

            height = output_shape[0] /  2**(num_layers - 1)
            width = output_shape[1] / 2**(num_layers - 1)
            h0 = ops.linear(opts, noise, num_filters * height * width,
                            scope='h0_lin')
            h0 = tf.reshape(h0, [-1, height, width, num_filters])
            h0 = tf.nn.relu(h0)

            layer_x = h0
            for i in xrange(num_layers-1):
                scale = 2**(i+1)
                _out_shape = [dim1, height * scale, width * scale, num_filters / scale]
                layer_x = ops.deconv2d(opts, layer_x, _out_shape, scope='h%d_deconv' % i)
                if opts['batch_norm']:
                    layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d' % i)
                layer_x = tf.nn.relu(layer_x)
                if opts['dropout']:
                    _keep_prob = tf.minimum(
                        1., 0.9 - (0.9 - keep_prob) * float(i + 1) / (num_layers - 1))
                    layer_x = tf.nn.dropout(layer_x, _keep_prob)

            # # h0 = ops.lrelu(h0)
            # _out_shape = [dim1, height * 2, width * 2, num_filters / 2]
            # # for 28 x 28 does 7 x 7 --> 14 x 14
            # h1 = ops.deconv2d(opts, h0, _out_shape, scope='h1_deconv')
            # h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            # h1 = tf.nn.relu(h1)
            # # h1 = ops.lrelu(h1)
            # _out_shape = [dim1, height * 4, width * 4, num_filters / 4]
            # # for 28 x 28 does 14 x 14 --> 28 x 28
            # h2 = ops.deconv2d(opts, h1, _out_shape, scope='h2_deconv')
            # h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            # h2 = tf.nn.relu(h2)
            # # h2 = ops.lrelu(h2)

            _out_shape = [dim1] + list(output_shape)
            # data_shape[0] x data_shape[1] x ? -> data_shape
            h3 = ops.deconv2d(opts, layer_x, _out_shape,
                              d_h=1, d_w=1, scope='hlast_deconv')
            # h3 = ops.batch_norm(opts, h3, is_training, reuse, scope='bn_layer4')

        if return_logits:
            return h3
        if opts['input_normalize_sym']:
            return tf.nn.tanh(h3)
        else:
            return tf.nn.sigmoid(h3)

    def discriminator(self, opts, input_, is_training,
                      prefix='DISCRIMINATOR', reuse=False):
        """Encoder function, suitable for simple toy experiments.

        """
        num_filters = opts['d_num_filters']

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.conv2d(opts, input_, num_filters / 8, scope='h0_conv')
            h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
            h0 = tf.nn.relu(h0)
            h1 = ops.conv2d(opts, h0, num_filters / 4, scope='h1_conv')
            h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            h1 = tf.nn.relu(h1)
            h2 = ops.conv2d(opts, h1, num_filters / 2, scope='h2_conv')
            h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            h2 = tf.nn.relu(h2)
            h3 = ops.conv2d(opts, h2, num_filters, scope='h3_conv')
            h3 = ops.batch_norm(opts, h3, is_training, reuse, scope='bn_layer4')
            h3 = tf.nn.relu(h3)
            # Already has NaNs!!
            latent_mean = ops.linear(opts, h3, opts['latent_space_dim'], scope='h3_lin')
            log_latent_sigmas = ops.linear(opts, h3, opts['latent_space_dim'], scope='h3_lin_sigma')

        return latent_mean, log_latent_sigmas

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to VAE implementation.

        """
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        is_training_ph = tf.placeholder(tf.bool, name='is_train_ph')
        lr_decay_ph = tf.placeholder(tf.float32)


        # Operations

        latent_x_mean, log_latent_sigmas = self.discriminator(
            opts, real_points_ph, is_training_ph)
        scaled_noise = tf.multiply(
            tf.sqrt(1e-6 + tf.exp(log_latent_sigmas)), noise_ph)
        loss_kl = 0.5 * tf.reduce_sum(
            tf.exp(log_latent_sigmas) +
            tf.square(latent_x_mean) -
            log_latent_sigmas, axis=1)
        if opts['recon_loss'] == 'l2sq':
            reconstruct_x = self.generator(opts, latent_x_mean + scaled_noise,
                                           is_training_ph)
            loss_reconstruct = tf.reduce_sum(
                tf.square(real_points_ph - reconstruct_x), axis=[1,2,3])
            loss_reconstruct = loss_reconstruct / 2. / opts['vae_sigma']
        elif opts['recon_loss'] == 'cross_entropy':
            if opts['input_normalize_sym']:
                expected = (real_points_ph + 1.0) / 2.0
            else:
                expected = real_points_ph
            reconstruct_x_logits = self.generator(
                opts, latent_x_mean + scaled_noise,
                is_training_ph, return_logits=True)
            loss_reconstruct = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=expected, logits=reconstruct_x_logits),
                axis=[1,2,3])
        else:
            raise ValueError("Unknown recon loss value %s" % opts['recon_loss'])
        dec_enc_x = self.generator(opts, latent_x_mean,
                                   is_training=False, reuse=True)

        loss_reconstruct = tf.reduce_mean(loss_reconstruct)
        loss_kl = tf.reduce_mean(loss_kl)
        loss = loss_kl + loss_reconstruct
        # loss = tf.Print(loss, [loss, loss_kl, loss_reconstruct], 'Loss, KL, reconstruct')
        optim = ops.optimizer(opts, decay=lr_decay_ph).minimize(loss)

        generated_images = self.generator(opts, noise_ph,
                                          is_training_ph, reuse=True)

        self._real_points_ph = real_points_ph
        self._noise_ph = noise_ph
        self._is_training_ph = is_training_ph
        self._optim = optim
        self._loss = loss
        self._loss_reconstruct = loss_reconstruct
        self._lr_decay_ph = lr_decay_ph
        self._loss_kl = loss_kl
        self._generated = generated_images
        self._reconstruct_x = dec_enc_x
        self._enc_mean = latent_x_mean
        self._enc_log_var = log_latent_sigmas

        saver = tf.train.Saver(max_to_keep=10)
        tf.add_to_collection('real_points_ph', self._real_points_ph)
        tf.add_to_collection('noise_ph', self._noise_ph)
        tf.add_to_collection('is_training_ph', self._is_training_ph)
        tf.add_to_collection('encoder_mean', self._enc_mean)
        tf.add_to_collection('encoder_log_sigma', self._enc_log_var)
        tf.add_to_collection('decoder', self._generated)

        self._saver = saver

        logging.error("Building Graph Done.")


    def _train_internal(self, opts):
        """Train a VAE model.

        """

        batches_num = self._data.num_points / opts['batch_size']
        train_size = self._data.num_points
        num_plot = 320
        sample_prev = np.zeros([num_plot] + list(self._data.data_shape))
        l2s = []

        counter = 0
        decay = 1.
        logging.error('Training VAE')
        for _epoch in xrange(opts["gan_epoch_num"]):

            if opts['decay_schedule'] == "manual":
                if _epoch == 30:
                    decay = decay / 2.
                if _epoch == 50:
                    decay = decay / 5.
                if _epoch == 100:
                    decay = decay / 10.

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
                batch_noise = utils.generate_noise(opts, opts['batch_size'])
                _, loss, loss_kl, loss_reconstruct = self._session.run(
                    [self._optim, self._loss, self._loss_kl,
                     self._loss_reconstruct],
                    feed_dict={self._real_points_ph: batch_images,
                               self._noise_ph: batch_noise,
                               self._lr_decay_ph: decay,
                               self._is_training_ph: True})
                counter += 1

                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    debug_str = 'Epoch: %d/%d, batch:%d/%d' % (
                        _epoch+1, opts['gan_epoch_num'], _idx+1, batches_num)
                    debug_str += '  [L=%.2g, Recon=%.2g, KLQ=%.2g]' % (
                        loss, loss_reconstruct, loss_kl)
                    logging.error(debug_str)

                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._generated, self._noise_ph,
                        self._noise_for_plots[0:num_plot],
                        self._is_training_ph, False)
                    l2s.append(np.sum((points_to_plot - sample_prev)**2))
                    metrics.l2s = l2s[:]
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        points_to_plot,
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))
                    reconstructed = self._session.run(
                        self._reconstruct_x,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._is_training_ph: False})
                    metrics.l2s = None
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        reconstructed,
                        prefix='reconstr_e%04d_mb%05d_' % (_epoch, _idx))
                if opts['early_stop'] > 0 and counter > opts['early_stop']:
                    break
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
        noise = utils.generate_noise(opts, num)
        sample = self._run_batch(
            opts, self._generated, self._noise_ph, noise,
            self._is_training_ph, False)
        return sample



