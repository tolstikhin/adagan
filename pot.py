# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class implements POT training.

"""

import logging
import os
import tensorflow as tf
import utils
from utils import ProgressBar
from utils import TQDM
import numpy as np
import ops
from metrics import Metrics

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
        self._noise_for_plots = 2 * utils.generate_noise(opts, 500)
        # Placeholders
        self._real_points_ph = None
        self._noise_ph = None

        # Main operations

        # Optimizers

        with self._session.as_default(), self._session.graph.as_default():
            logging.debug('Building the graph...')
            self._build_model_internal(opts)

        # Make sure AdamOptimizer, if used in the Graph, is defined before
        # calling global_variables_initializer().
        init = tf.global_variables_initializer()
        self._session.run(init)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleaning the whole default Graph
        logging.debug('Cleaning the graph...')
        tf.reset_default_graph()
        logging.debug('Closing the session...')
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

    def generator(self, opts, noise, reuse=False):
        """ Decoder actually.

        """

        output_shape = self._data.data_shape
        num_units = opts['g_num_filters']

        with tf.variable_scope("GENERATOR", reuse=reuse):
            if not opts['convolutions']:
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
            else:
                dim1 = tf.shape(noise)[0]
                height = output_shape[0] / 4
                width = output_shape[1] / 4

                h0 = ops.linear(
                    opts, noise, num_units * height * width, scope='h0_lin')
                h0 = tf.reshape(h0, [-1, height, width, num_units])
                h0 = tf.nn.relu(h0)
                _out_shape = [dim1, height * 2, width * 2, num_units / 2]
                h1 = ops.deconv2d(opts, h0, _out_shape, scope='h1_deconv')
                h1 = tf.nn.relu(h1)
                _out_shape = [dim1, height * 4, width * 4, num_units / 4]
                h2 = ops.deconv2d(opts, h1, _out_shape, scope='h2_deconv')
                h2 = tf.nn.relu(h2)
                _out_shape = [dim1] + list(output_shape)
                h3 = ops.deconv2d(
                    opts, h2, _out_shape, d_h=1, d_w=1, scope='h3_deconv')
                if opts['input_normalize_sym']:
                    return tf.nn.tanh(h3)
                else:
                    return tf.nn.sigmoid(h3)


    # def generator(self, opts, noise, is_training, reuse=False):

    #     output_shape = self._data.data_shape # (dim1, dim2, dim3)
    #     # Computing the number of noise vectors on-the-go
    #     dim1 = tf.shape(noise)[0]
    #     num_filters = opts['g_num_filters']

    #     with tf.variable_scope("GENERATOR", reuse=reuse):

    #         height = output_shape[0] / 4
    #         width = output_shape[1] / 4
    #         h0 = ops.linear(opts, noise, num_filters * height * width,
    #                         scope='h0_lin')
    #         h0 = tf.reshape(h0, [-1, height, width, num_filters])
    #         h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
    #         # h0 = tf.nn.relu(h0)
    #         h0 = ops.lrelu(h0)
    #         _out_shape = [dim1, height * 2, width * 2, num_filters / 2]
    #         # for 28 x 28 does 7 x 7 --> 14 x 14
    #         h1 = ops.deconv2d(opts, h0, _out_shape, scope='h1_deconv')
    #         h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
    #         # h1 = tf.nn.relu(h1)
    #         h1 = ops.lrelu(h1)
    #         _out_shape = [dim1, height * 4, width * 4, num_filters / 4]
    #         # for 28 x 28 does 14 x 14 --> 28 x 28
    #         h2 = ops.deconv2d(opts, h1, _out_shape, scope='h2_deconv')
    #         h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
    #         # h2 = tf.nn.relu(h2)
    #         h2 = ops.lrelu(h2)
    #         _out_shape = [dim1] + list(output_shape)
    #         # data_shape[0] x data_shape[1] x ? -> data_shape
    #         h3 = ops.deconv2d(opts, h2, _out_shape,
    #                           d_h=1, d_w=1, scope='h3_deconv')
    #         h3 = ops.batch_norm(opts, h3, is_training, reuse, scope='bn_layer4')

    #     if opts['input_normalize_sym']:
    #         return tf.nn.tanh(h3)
    #     else:
    #         return tf.nn.sigmoid(h3)

    def discriminator(self, opts, input_, prefix='DISCRIMINATOR', reuse=False):
        """Discriminator for the GAN objective

        """

        num_units = opts['d_num_filters']
        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.linear(opts, input_, num_units, scope='h0_lin')
            # h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
            h0 = tf.nn.relu(h0)
            h1 = ops.linear(opts, h0, num_units, scope='h1_lin')
            # h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            h1 = tf.nn.relu(h1)
            h2 = ops.linear(opts, h1, num_units, scope='h2_lin')
            h2 = tf.nn.relu(h2)
            h3 = ops.linear(opts, h2, 1, scope='h3_lin')

        return h3

    def encoder(self, opts, input_, reuse=False):

        num_units = opts['g_num_filters']
        with tf.variable_scope("ENCODER", reuse=reuse):
            if not opts['convolutions']:
                h0 = ops.linear(opts, input_, 1024, 'h0_lin')
                h0 = tf.nn.relu(h0)
                h1 = ops.linear(opts, h0, 512, 'h1_lin')
                h1 = tf.nn.relu(h1)
                h2 = ops.linear(opts, h1, 512, 'h2_lin')
                h2 = tf.nn.relu(h2)
                code = ops.linear(opts, h2, opts['latent_space_dim'], 'h3_lin')
            else:
                h0 = ops.conv2d(opts, input_, num_units, scope='h0_conv')
                h0 = tf.nn.relu(h0)
                h1 = ops.conv2d(opts, h0, num_units * 2, scope='h1_conv')
                h1 = tf.nn.relu(h1)
                h2 = ops.conv2d(opts, h1, num_units * 4, scope='h2_conv')
                h2 = tf.nn.relu(h2)
                code = ops.linear(opts, h2, opts['latent_space_dim'], scope='h3_lin')

        return code

    # def encoder(self, opts, input_,
    #                   prefix='ENCODER', reuse=False):

    #     num_filters = 32
    #     with tf.variable_scope(prefix, reuse=reuse):
    #         h0 = ops.conv2d(opts, input_, num_filters, scope='h0_conv')
    #         #h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
    #         h0 = tf.nn.relu(h0)
    #         h1 = ops.conv2d(opts, h0, num_filters * 2, scope='h1_conv')
    #         #h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
    #         h1 = tf.nn.relu(h1)
    #         h2 = ops.conv2d(opts, h1, num_filters * 4, scope='h2_conv')
    #         #h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
    #         h2 = tf.nn.relu(h2)
    #         code = ops.linear(opts, h2, opts['latent_space_dim'], scope='h3_lin')

    #     return code

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to POT implementation.

        """
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')

        # Operations

        encoded_training = self.encoder(opts, real_points_ph)
        reconstructed_training = self.generator(opts, encoded_training)

        # c(x,y) = ||x - y||
        loss_reconstr = tf.reduce_sum(
            tf.square(real_points_ph - reconstructed_training), axis=1)
        # sqrt(x + delta) guarantees the direvative 1/(x + delta) is finite
        loss_reconstr = tf.reduce_mean(tf.sqrt(loss_reconstr + 1e-08))

        d_logits_Pz = self.discriminator(opts, noise_ph)
        d_logits_Qz = self.discriminator(opts, encoded_training, reuse=True)
        d_loss_Pz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_Pz, labels=tf.ones_like(d_logits_Pz)))
        d_loss_Qz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_Qz, labels=tf.zeros_like(d_logits_Qz)))
        d_loss = opts['pot_lambda'] * (d_loss_Pz + d_loss_Qz)
        d_loss = tf.Print(d_loss,
                          [tf.reduce_mean(d_logits_Pz, axis=0),
                           tf.reduce_mean(d_logits_Qz, axis=0)],
                          'D(Pz), D(Qz): ')

        loss_gan = - tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_Qz, labels=tf.zeros_like(d_logits_Qz)))

        loss = loss_reconstr + opts['pot_lambda'] * loss_gan
        loss = tf.Print(loss, [loss, loss_reconstr, loss_gan], 'loss, reconstruct, gan')


        t_vars = tf.trainable_variables()
        # Updates for discriminator
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        # Updates for encoder and generator
        eg_vars = [var for var in t_vars if 'DISCRIMINATOR/' not in var.name]

        d_optim = ops.optimizer(opts, 'd').minimize(loss=d_loss, var_list=d_vars)
        optim = ops.optimizer(opts, 'g').minimize(loss=loss, var_list=eg_vars)

        generated_images = self.generator(opts, noise_ph, reuse=True)

        self._real_points_ph = real_points_ph
        self._noise_ph = noise_ph
        self._optim = optim
        self._d_optim = d_optim
        self._loss = loss
        self._loss_reconstruct = loss_reconstr
        self._loss_gan = loss_gan
        self._d_loss = d_loss
        self._generated = generated_images
        self._Qz = encoded_training
        self._reconstruct_x = reconstructed_training

        saver = tf.train.Saver()
        tf.add_to_collection('real_points_ph', self._real_points_ph)
        tf.add_to_collection('noise_ph', self._noise_ph)
        tf.add_to_collection('encoder', self._Qz)
        tf.add_to_collection('decoder', self._generated)
        tf.add_to_collection('disc_logits_Pz', d_logits_Pz)
        tf.add_to_collection('disc_logits_Qz', d_logits_Qz)

        self._saver = saver

        logging.debug("Building Graph Done.")


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
        logging.debug('Training POT')
        for _epoch in xrange(opts["gan_epoch_num"]):
            for _idx in xrange(batches_num):
                # logging.debug('Step %d of %d' % (_idx, batches_num ) )
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=self._data_weights)
                batch_images = self._data.data[data_ids].astype(np.float)
                # batch_labels = self._data.labels[data_ids].astype(np.int32)
                batch_noise = 2 * utils.generate_noise(opts, opts['batch_size'])

                # Update generator and encoder
                for _ in range(1):
                    [_, loss, loss_rec, loss_gan] = self._session.run(
                        [self._optim,
                         self._loss,
                         self._loss_reconstruct,
                         self._loss_gan],
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise})
                losses.append(loss)

                # Update discriminator
                _ = self._session.run(
                    [self._d_optim, self._d_loss],
                    feed_dict={self._real_points_ph: batch_images,
                               self._noise_ph: batch_noise})
                counter += 1


                if counter > 0 and counter % opts['save_every'] == 0:
                    os.path.join(opts['work_dir'], opts['ckpt_dir'])
                    self._saver.save(self._session,
                                     os.path.join(opts['work_dir'],
                                                  opts['ckpt_dir'],
                                                  'trained-pot'),
                                     global_step=counter)

                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    logging.debug(
                        'Epoch: %d/%d, batch:%d/%d' % \
                        (_epoch+1, opts['gan_epoch_num'], _idx+1, batches_num))
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._generated, self._noise_ph,
                        self._noise_for_plots[0:num_plot])
                    Qz_sample = self._run_batch(
                            opts, self._Qz, self._real_points_ph, self._data.data[:1000])
                    metrics.Qz = Qz_sample
                    metrics.Qz_labels = self._data.labels[:1000]
                    metrics.Pz = batch_noise
                    l2s.append(np.sum((points_to_plot - sample_prev)**2))
                    # metrics.l2s = l2s[:]
                    metrics.l2s = losses[:]
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        np.vstack([points_to_plot, 0 * batch_images[:16], batch_images]),
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))
                    reconstructed = self._session.run(
                        self._reconstruct_x,
                        feed_dict={self._real_points_ph: batch_images})
                    # metrics.l2s = None
                    # metrics.Qz = None
                    # metrics.Pz = None
                    merged = np.vstack([reconstructed, batch_images])
                    r_ptr = 0
                    w_ptr = 0
                    for _ in range(opts['batch_size']):
                        merged[w_ptr] = batch_images[r_ptr]
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
                if opts['early_stop'] > 0 and counter > opts['early_stop']:
                    break

    def _sample_internal(self, opts, num):
        """Sample from the trained GAN model.

        """

        assert False, 'Need to code'



