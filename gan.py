# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class implements Generative Adversarial Networks training.

"""

import logging
import tensorflow as tf
import utils
from utils import ProgressBar
from utils import TQDM
import numpy as np
import ops
from metrics import Metrics

class Gan(object):
    """A base class for running individual GANs.

    This class announces all the necessary bits for running individual
    GAN trainers. It is assumed that a GAN trainer should receive the
    data points and the corresponding weights, which are used for
    importance sampling of minibatches during the training. All the
    methods should be implemented in the subclasses.
    """
    def __init__(self, opts, data, weights):

        # Create a new session with session.graph = default graph
        self._session = tf.Session()
        self._trained = False
        self._data = data
        self._data_weights = np.copy(weights)
        # Latent noise sampled ones to apply G while training
        self._noise_for_plots = utils.generate_noise(opts, 500)
        # Placeholders
        self._real_points_ph = None
        self._fake_points_ph = None
        self._noise_ph = None
        self._inv_target_ph = None

        # Main operations
        self._G = None # Generator function
        self._d_loss = None # Loss of discriminator
        self._g_loss = None # Loss of generator
        self._c_loss = None # Loss of mixture discriminator
        self._c_training = None # Outputs of the mixture discriminator on data
        self._inv_loss = None
        self._inv_loss_per_point = None

        # Variables
        self._inv_z = None

        # Optimizers
        self._g_optim = None
        self._d_optim = None
        self._c_optim = None
        self._inv_optim = None

        with self._session.as_default(), self._session.graph.as_default():
            logging.debug('Building the graph...')
            self._build_model_internal(opts)
            if opts['inverse_metric']:
                assert opts['dataset'] in ('mnist', 'mnist3', 'guitars'),\
                    'Invertion currently supported only for mnist, mnist3, guitars'
                logging.debug('Adding inversion ops to the graph...')
                self._add_inversion_ops(opts)

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
        """Train a GAN model.

        """
        with self._session.as_default(), self._session.graph.as_default():
            self._train_internal(opts)
            self._trained = True

    def sample(self, opts, num=100):
        """Sample points from the trained GAN model.

        """
        assert self._trained, 'Can not sample from the un-trained GAN'
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

    def invert_points(self, opts, images):
        """Invert the learned generator function for every image in images.

        Args:
            images: numpy array of shape [num_points] + data_shape

        """
        assert self._trained, 'Can not invert, not trained yet.'
        assert len(images) == opts['inverse_num'],\
            'Currently inversion works only for fixed number of images'
        with self._session.as_default(), self._session.graph.as_default():
            target_ph = self._inv_target_ph
            z = self._inv_z
            loss_per_point = self._inv_loss_per_point
            optim = self._inv_optim
            norms = self._inv_norms

            val_list = []
            err_per_point_list = []
            z_list = []
            norms_list = []
            for _start in xrange(5):
                inv_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="inversion")
                # Initialize z and optimizer's variables randomly
                self._session.run(tf.variables_initializer(inv_vars))
                prev_val = 100.
                check_every = 100
                steps = 1
                while True:
                    # Stopping criterion: relative improvement of the maximal
                    # per point mse gets smaller than a threshold
                    self._session.run(
                        optim, feed_dict={target_ph:images})
                    if steps % check_every == 0:
                        err_per_point = loss_per_point.eval(
                            feed_dict={target_ph:images})
                        err_max = np.max(err_per_point)
                        err = np.mean(err_per_point)
                        logging.debug('Init %02d, steps %d, loss %f, max mse %f' %\
                                      (_start, steps, err, err_max))
                        relative_improvement = np.abs(prev_val - err) / prev_val
                        if relative_improvement < 1e-3 or steps > 10000:
                            val_list.append(err)
                            err_per_point_list.append(err_per_point)
                            z_list.append(self._session.run(z))
                            norms_list.append(self._session.run(norms))
                            break
                        prev_val = err
                    steps += 1
            # Choose the run where we got the best (i.e. minimal) maximal
            # per point mse
            best_id = sorted(zip(val_list, range(len(val_list))))[0][1]
            best_err_per_point = err_per_point_list[best_id]
            best_z = z_list[best_id]
            best_norms = norms_list[best_id]
            best_reconstructions = self._G.eval(
                feed_dict={self._noise_ph:best_z,
                           self._is_training_ph:False})

            return best_reconstructions, best_z, best_err_per_point, best_norms

    def _add_inversion_ops(self, opts):
        data_shape = self._data.data_shape
        with tf.variable_scope("inversion"):
            target_ph = tf.placeholder(
                tf.float32, [None] + list(data_shape),
                name='target_ph')
            z = tf.get_variable(
                "inverted", [opts['inverse_num'], opts['latent_space_dim']],
                tf.float32, tf.random_normal_initializer(stddev=1.))
        reconstructed_images = self.generator(
            opts, z, is_training=False, reuse=True)
        with tf.variable_scope("inversion"):
            loss_per_point = tf.reduce_mean(
                tf.square(tf.subtract(reconstructed_images, target_ph)),
                axis=[1, 2, 3])
            loss = tf.reduce_mean(loss_per_point)
            norms = tf.reduce_sum(tf.square(z), axis=[1])
            optim = tf.train.AdamOptimizer(0.01, 0.9)
            optim = optim.minimize(loss, var_list=[z])

        self._inv_target_ph = target_ph
        self._inv_z = z
        self._inv_optim = optim
        self._inv_loss = loss
        self._inv_loss_per_point = loss_per_point
        self._inv_norms = norms

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
        assert False, 'Gan base class has no build_model method defined.'

    def _train_internal(self, opts):
        assert False, 'Gan base class has no train method defined.'

    def _sample_internal(self, opts, num):
        assert False, 'Gan base class has no sample method defined.'

    def _train_mixture_discriminator_internal(self, opts, fake_images):
        assert False, 'Gan base class has no mixture discriminator method defined.'

class ToyGan(Gan):
    """A simple GAN implementation, suitable for toy datasets.

    """

    def generator(self, opts, noise, reuse=False):
        """Generator function, suitable for simple toy experiments.

        Args:
            noise: [num_points, dim] array, where dim is dimensionality of the
                latent noise space.
        Returns:
            [num_points, dim1, dim2, dim3] array, where the first coordinate
            indexes the points, which all are of the shape (dim1, dim2, dim3).
        """
        output_shape = self._data.data_shape

        with tf.variable_scope("GENERATOR", reuse=reuse):
            h0 = ops.linear(opts, noise, 500, 'h0_lin')
            h0 = tf.nn.relu(h0)
            h1 = ops.linear(opts, h0, 500, 'h1_lin')
            h1 = tf.nn.relu(h1)
            h2 = ops.linear(opts, h1, np.prod(output_shape), 'h2_lin')
            h2 = tf.reshape(h2, [-1] + list(output_shape))

        if opts['input_normalize_sym']:
            return tf.nn.tanh(h2)
        else:
            return tf.nn.sigmoid(h2)

    def discriminator(self, opts, input_,
                      prefix='DISCRIMINATOR', reuse=False):
        """Discriminator function, suitable for simple toy experiments.

        """
        shape = input_.get_shape().as_list()
        assert len(shape) > 0, 'No inputs to discriminate.'

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.linear(opts, input_, 500, 'h0_lin')
            h0 = tf.nn.relu(h0)
            h1 = ops.linear(opts, h0, 500, 'h1_lin')
            h1 = tf.nn.relu(h1)
            h2 = ops.linear(opts, h1, 1, 'h2_lin')

        return h2

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to GAN implementation.

        """
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')

        # Operations
        G = self.generator(opts, noise_ph)

        d_logits_real = self.discriminator(opts, real_points_ph)
        d_logits_fake = self.discriminator(opts, G, reuse=True)

        c_logits_real = self.discriminator(
            opts, real_points_ph, prefix='CLASSIFIER')
        c_logits_fake = self.discriminator(
            opts, fake_points_ph, prefix='CLASSIFIER', reuse=True)
        c_training = tf.nn.sigmoid(
            self.discriminator(opts, real_points_ph, prefix='CLASSIFIER', reuse=True))

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_real, labels=tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_fake, labels=tf.zeros_like(c_logits_fake)))
        c_loss = c_loss_real + c_loss_fake

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        g_vars = [var for var in t_vars if 'GENERATOR/' in var.name]
        d_optim = ops.optimizer(opts, 'd').minimize(d_loss, var_list=d_vars)
        g_optim = ops.optimizer(opts, 'g').minimize(g_loss, var_list=g_vars)
        c_vars = [var for var in t_vars if 'CLASSIFIER/' in var.name]
        c_optim = ops.optimizer(opts).minimize(c_loss, var_list=c_vars)

        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph

        self._G = G
        self._d_loss = d_loss
        self._g_loss = g_loss
        self._c_loss = c_loss
        self._c_training = c_training
        self._g_optim = g_optim
        self._d_optim = d_optim
        self._c_optim = c_optim

    def _train_internal(self, opts):
        """Train a GAN model.

        """

        batches_num = self._data.num_points / opts['batch_size']
        train_size = self._data.num_points

        counter = 0
        logging.debug('Training GAN')
        for _epoch in xrange(opts["gan_epoch_num"]):
            for _idx in xrange(batches_num):
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=self._data_weights)
                batch_images = self._data.data[data_ids].astype(np.float)
                batch_noise = utils.generate_noise(opts, opts['batch_size'])
                # Update discriminator parameters
                for _iter in xrange(opts['d_steps']):
                    _ = self._session.run(
                        self._d_optim,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise})
                # Update generator parameters
                for _iter in xrange(opts['g_steps']):
                    _ = self._session.run(
                        self._g_optim, feed_dict={self._noise_ph: batch_noise})
                counter += 1
                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._G, self._noise_ph,
                        self._noise_for_plots[0:320])
                    data_ids = np.random.choice(train_size, 320,
                                                replace=False,
                                                p=self._data_weights)
                    metrics.make_plots(
                        opts, counter,
                        self._data.data[data_ids],
                        points_to_plot,
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))



    def _sample_internal(self, opts, num):
        """Sample from the trained GAN model.

        """
        noise = utils.generate_noise(opts, num)
        sample = self._run_batch(opts, self._G, self._noise_ph, noise)
        # sample = self._session.run(
        #     self._G, feed_dict={self._noise_ph: noise})
        return sample

    def _train_mixture_discriminator_internal(self, opts, fake_images):
        """Train a classifier separating true data from points in fake_images.

        """

        batches_num = self._data.num_points / opts['batch_size']
        logging.debug('Training a mixture discriminator')
        for epoch in xrange(opts["mixture_c_epoch_num"]):
            for idx in xrange(batches_num):
                ids = np.random.choice(len(fake_images), opts['batch_size'],
                                       replace=False)
                batch_fake_images = fake_images[ids]
                ids = np.random.choice(self._data.num_points, opts['batch_size'],
                                       replace=False)
                batch_real_images = self._data.data[ids]
                _ = self._session.run(
                    self._c_optim,
                    feed_dict={self._real_points_ph: batch_real_images,
                               self._fake_points_ph: batch_fake_images})

        res = self._run_batch(
            opts, self._c_training,
            self._real_points_ph, self._data.data)
        return res, None


class ToyUnrolledGan(ToyGan):
    """A simple GAN implementation, suitable for toy datasets.

    """

    def __init__(self, opts, data, weights):

        # Losses of the copied discriminator network
        self._d_loss_cp = None
        self._d_optim_cp = None
        # Rolling back ops (assign variable values fo true
        # to copied discriminator network)
        self._roll_back = None

        Gan.__init__(self, opts, data, weights)

    # Architecture used in unrolled gan paper
    def generator(self, opts, noise, reuse=False):
        """Generator function, suitable for simple toy experiments.

        Args:
            noise: [num_points, dim] array, where dim is dimensionality of the
                latent noise space.
        Returns:
            [num_points, dim1, dim2, dim3] array, where the first coordinate
            indexes the points, which all are of the shape (dim1, dim2, dim3).
        """
        output_shape = self._data.data_shape

        with tf.variable_scope("GENERATOR", reuse=reuse):
            h0 = ops.linear(opts, noise, 500, 'h0_lin')
            h0 = tf.nn.tanh(h0)
            h1 = ops.linear(opts, h0, 500, 'h1_lin')
            h1 = tf.nn.tanh(h1)
            h2 = ops.linear(opts, h1, np.prod(output_shape), 'h2_lin')
            h2 = tf.reshape(h2, [-1] + list(output_shape))

        if opts['input_normalize_sym']:
            return tf.nn.tanh(h2)
        else:
            return tf.nn.sigmoid(h2)

    def discriminator(self, opts, input_,
                      prefix='DISCRIMINATOR', reuse=False):
        """Discriminator function, suitable for simple toy experiments.

        """
        shape = input_.get_shape().as_list()
        assert len(shape) > 0, 'No inputs to discriminate.'

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.linear(opts, input_, 500, 'h0_lin')
            h0 = tf.nn.tanh(h0)
            h1 = ops.linear(opts, h0, 500, 'h1_lin')
            h1 = tf.nn.tanh(h1)
            h2 = ops.linear(opts, h1, 1, 'h2_lin')

        return h2

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to GAN implementation.

        """
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')

        # Operations
        G = self.generator(opts, noise_ph)

        d_logits_real = self.discriminator(opts, real_points_ph)
        d_logits_fake = self.discriminator(opts, G, reuse=True)

        # Disccriminator copy for the unrolling steps
        d_logits_real_cp = self.discriminator(
            opts, real_points_ph, prefix='DISCRIMINATOR_CP')
        d_logits_fake_cp = self.discriminator(
            opts, G, prefix='DISCRIMINATOR_CP', reuse=True)

        c_logits_real = self.discriminator(
            opts, real_points_ph, prefix='CLASSIFIER')
        c_logits_fake = self.discriminator(
            opts, fake_points_ph, prefix='CLASSIFIER', reuse=True)
        c_training = tf.nn.sigmoid(
            self.discriminator(opts, real_points_ph, prefix='CLASSIFIER', reuse=True))

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        d_loss_real_cp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_real_cp, labels=tf.ones_like(d_logits_real_cp)))
        d_loss_fake_cp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake_cp,
                labels=tf.zeros_like(d_logits_fake_cp)))
        d_loss_cp = d_loss_real_cp + d_loss_fake_cp

        if opts['objective'] == 'JS':
            g_loss = - d_loss_cp
        elif opts['objective'] == 'JS_modified':
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logits_fake_cp, labels=tf.ones_like(d_logits_fake_cp)))
        else:
            assert False, 'No objective %r implemented' % opts['objective']

        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_real, labels=tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_fake, labels=tf.zeros_like(c_logits_fake)))
        c_loss = c_loss_real + c_loss_fake

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        d_vars_cp = [var for var in t_vars if 'DISCRIMINATOR_CP/' in var.name]
        c_vars = [var for var in t_vars if 'CLASSIFIER/' in var.name]
        g_vars = [var for var in t_vars if 'GENERATOR/' in var.name]

        # Ops to roll back the variable values of discriminator_cp
        # Will be executed each time before the unrolling steps
        with tf.variable_scope('assign'):
            roll_back = []
            for var, var_cp in zip(d_vars, d_vars_cp):
                roll_back.append(tf.assign(var_cp, var))

        d_optim = ops.optimizer(opts, 'd').minimize(d_loss, var_list=d_vars)
        d_optim_cp = ops.optimizer(opts, 'd').minimize(
           d_loss_cp,
           var_list=d_vars_cp)
        c_optim = ops.optimizer(opts).minimize(c_loss, var_list=c_vars)
        g_optim = ops.optimizer(opts, 'g').minimize(g_loss, var_list=g_vars)

        # writer = tf.summary.FileWriter(opts['work_dir']+'/tensorboard', self._session.graph)

        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph

        self._G = G
        self._roll_back = roll_back
        self._d_loss = d_loss
        self._d_loss_cp = d_loss_cp
        self._g_loss = g_loss
        self._c_loss = c_loss
        self._c_training = c_training
        self._g_optim = g_optim
        self._d_optim = d_optim
        self._d_optim_cp = d_optim_cp
        self._c_optim = c_optim

        logging.debug("Building Graph Done.")


    def _train_internal(self, opts):
        """Train a GAN model.

        """

        batches_num = self._data.num_points / opts['batch_size']
        train_size = self._data.num_points

        counter = 0
        logging.debug('Training GAN')
        for _epoch in xrange(opts["gan_epoch_num"]):
            for _idx in TQDM(opts, xrange(batches_num),
                             desc='Epoch %2d/%2d' %\
                             (_epoch+1, opts["gan_epoch_num"])):
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=self._data_weights)
                batch_images = self._data.data[data_ids].astype(np.float)
                batch_noise = utils.generate_noise(opts, opts['batch_size'])
                # Update discriminator parameters
                for _iter in xrange(opts['d_steps']):
                    _ = self._session.run(
                        self._d_optim,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise})
                # Roll back discriminator_cp's variables
                self._session.run(self._roll_back)
                # Unrolling steps
                for _iter in xrange(opts['unrolling_steps']):
                    self._session.run(
                        self._d_optim_cp,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise})
                # Update generator parameters
                for _iter in xrange(opts['g_steps']):
                    _ = self._session.run(
                        self._g_optim, feed_dict={self._noise_ph: batch_noise})
                counter += 1
                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._G, self._noise_ph,
                        self._noise_for_plots[0:320])
                    data_ids = np.random.choice(train_size, 320,
                                                replace=False,
                                                p=self._data_weights)
                    metrics.make_plots(
                        opts, counter,
                        self._data.data[data_ids],
                        points_to_plot,
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))

class ImageGan(Gan):
    """A simple GAN implementation, suitable for pictures.

    """

    def __init__(self, opts, data, weights):

        # One more placeholder for batch norm
        self._is_training_ph = None

        Gan.__init__(self, opts, data, weights)

    def generator(self, opts, noise, is_training, reuse=False):
        """Generator function, suitable for simple picture experiments.

        Args:
            noise: [num_points, dim] array, where dim is dimensionality of the
                latent noise space.
            is_training: bool, defines whether to use batch_norm in the train
                or test mode.
        Returns:
            [num_points, dim1, dim2, dim3] array, where the first coordinate
            indexes the points, which all are of the shape (dim1, dim2, dim3).
        """

        output_shape = self._data.data_shape # (dim1, dim2, dim3)
        # Computing the number of noise vectors on-the-go
        dim1 = tf.shape(noise)[0]
        num_filters = opts['g_num_filters']

        with tf.variable_scope("GENERATOR", reuse=reuse):

            height = output_shape[0] / 4
            width = output_shape[1] / 4
            h0 = ops.linear(opts, noise, num_filters * height * width,
                            scope='h0_lin')
            h0 = tf.reshape(h0, [-1, height, width, num_filters])
            h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
            # h0 = tf.nn.relu(h0)
            h0 = ops.lrelu(h0)
            _out_shape = [dim1, height * 2, width * 2, num_filters / 2]
            # for 28 x 28 does 7 x 7 --> 14 x 14
            h1 = ops.deconv2d(opts, h0, _out_shape, scope='h1_deconv')
            h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            # h1 = tf.nn.relu(h1)
            h1 = ops.lrelu(h1)
            _out_shape = [dim1, height * 4, width * 4, num_filters / 4]
            # for 28 x 28 does 14 x 14 --> 28 x 28
            h2 = ops.deconv2d(opts, h1, _out_shape, scope='h2_deconv')
            h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            # h2 = tf.nn.relu(h2)
            h2 = ops.lrelu(h2)
            _out_shape = [dim1] + list(output_shape)
            # data_shape[0] x data_shape[1] x ? -> data_shape
            h3 = ops.deconv2d(opts, h2, _out_shape,
                              d_h=1, d_w=1, scope='h3_deconv')
            h3 = ops.batch_norm(opts, h3, is_training, reuse, scope='bn_layer4')

        if opts['input_normalize_sym']:
            return tf.nn.tanh(h3)
        else:
            return tf.nn.sigmoid(h3)

    def discriminator(self, opts, input_, is_training,
                      prefix='DISCRIMINATOR', reuse=False):
        """Discriminator function, suitable for simple toy experiments.

        """
        num_filters = opts['d_num_filters']

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.conv2d(opts, input_, num_filters, scope='h0_conv')
            h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
            h0 = ops.lrelu(h0)
            h1 = ops.conv2d(opts, h0, num_filters * 2, scope='h1_conv')
            h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            h1 = ops.lrelu(h1)
            h2 = ops.conv2d(opts, h1, num_filters * 4, scope='h2_conv')
            h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            h2 = ops.lrelu(h2)
            h3 = ops.linear(opts, h2, 1, scope='h3_lin')

        return h3

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to GAN implementation.

        """
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        is_training_ph = tf.placeholder(tf.bool, name='is_train_ph')


        # Operations
        G = self.generator(opts, noise_ph, is_training_ph)
        # We use conv2d_transpose in the generator, which results in the
        # output tensor of undefined shapes. However, we statically know
        # the shape of the generator output, which is [-1, dim1, dim2, dim3]
        # where (dim1, dim2, dim3) is given by self._data.data_shape
        G.set_shape([None] + list(self._data.data_shape))

        d_logits_real = self.discriminator(opts, real_points_ph, is_training_ph)
        d_logits_fake = self.discriminator(opts, G, is_training_ph, reuse=True)

        c_logits_real = self.discriminator(
            opts, real_points_ph, is_training_ph, prefix='CLASSIFIER')
        c_logits_fake = self.discriminator(
            opts, fake_points_ph, is_training_ph, prefix='CLASSIFIER', reuse=True)
        c_training = tf.nn.sigmoid(
            self.discriminator(opts, real_points_ph, is_training_ph,
                               prefix='CLASSIFIER', reuse=True))

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_real, labels=tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_fake, labels=tf.zeros_like(c_logits_fake)))
        c_loss = c_loss_real + c_loss_fake

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        g_vars = [var for var in t_vars if 'GENERATOR/' in var.name]

        d_optim = ops.optimizer(opts, 'd').minimize(d_loss, var_list=d_vars)
        g_optim = ops.optimizer(opts, 'g').minimize(g_loss, var_list=g_vars)

        # d_optim_op = ops.optimizer(opts, 'd')
        # g_optim_op = ops.optimizer(opts, 'g')

        # def debug_grads(grad, var):
        #     _grad =  tf.Print(
        #         grad, # grads_and_vars,
        #         [tf.global_norm([grad])], 
        #         'Global grad norm of %s: ' % var.name)
        #     return _grad, var

        # d_grads_and_vars = [debug_grads(grad, var) for (grad, var) in \
        #     d_optim_op.compute_gradients(d_loss, var_list=d_vars)]
        # g_grads_and_vars = [debug_grads(grad, var) for (grad, var) in \
        #     g_optim_op.compute_gradients(g_loss, var_list=g_vars)]
        # d_optim = d_optim_op.apply_gradients(d_grads_and_vars)
        # g_optim = g_optim_op.apply_gradients(g_grads_and_vars)

        c_vars = [var for var in t_vars if 'CLASSIFIER/' in var.name]
        c_optim = ops.optimizer(opts).minimize(c_loss, var_list=c_vars)

        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph
        self._is_training_ph = is_training_ph
        self._G = G
        self._d_loss = d_loss
        self._g_loss = g_loss
        self._c_loss = c_loss
        self._c_training = c_training
        self._g_optim = g_optim
        self._d_optim = d_optim
        self._c_optim = c_optim

        logging.debug("Building Graph Done.")


    def _train_internal(self, opts):
        """Train a GAN model.

        """

        batches_num = self._data.num_points / opts['batch_size']
        train_size = self._data.num_points

        counter = 0
        logging.debug('Training GAN')
        for _epoch in xrange(opts["gan_epoch_num"]):
            for _idx in xrange(batches_num):
                # logging.debug('Step %d of %d' % (_idx, batches_num ) )
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=self._data_weights)
                batch_images = self._data.data[data_ids].astype(np.float)
                batch_noise = utils.generate_noise(opts, opts['batch_size'])
                # Update discriminator parameters
                for _iter in xrange(opts['d_steps']):
                    _ = self._session.run(
                        self._d_optim,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                # Update generator parameters
                for _iter in xrange(opts['g_steps']):
                    _ = self._session.run(
                        self._g_optim,
                        feed_dict={self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                counter += 1

                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    logging.debug(
                        'Epoch: %d/%d, batch:%d/%d' % \
                        (_epoch+1, opts['gan_epoch_num'], _idx+1, batches_num))
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._G, self._noise_ph,
                        self._noise_for_plots[0:320],
                        self._is_training_ph, False)
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        points_to_plot,
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))
                if opts['early_stop'] > 0 and counter > opts['early_stop']:
                    break

    def _sample_internal(self, opts, num):
        """Sample from the trained GAN model.

        """
        noise = utils.generate_noise(opts, num)
        sample = self._run_batch(
            opts, self._G, self._noise_ph, noise,
            self._is_training_ph, False)
        # sample = self._session.run(
        #     self._G, feed_dict={self._noise_ph: noise})
        return sample

    def _train_mixture_discriminator_internal(self, opts, fake_images):
        """Train a classifier separating true data from points in fake_images.

        """

        batches_num = self._data.num_points / opts['batch_size']
        logging.debug('Training a mixture discriminator')
        logging.debug('Using %d real points and %d fake ones' %\
                      (self._data.num_points, len(fake_images)))
        for epoch in xrange(opts["mixture_c_epoch_num"]):
            for idx in xrange(batches_num):
                ids = np.random.choice(len(fake_images), opts['batch_size'],
                                       replace=False)
                batch_fake_images = fake_images[ids]
                ids = np.random.choice(self._data.num_points, opts['batch_size'],
                                       replace=False)
                batch_real_images = self._data.data[ids]
                _ = self._session.run(
                    self._c_optim,
                    feed_dict={self._real_points_ph: batch_real_images,
                               self._fake_points_ph: batch_fake_images,
                               self._is_training_ph: True})

        # Evaluating trained classifier on real points
        res = self._run_batch(
            opts, self._c_training,
            self._real_points_ph, self._data.data,
            self._is_training_ph, False)

        # Evaluating trained classifier on fake points
        res_fake = self._run_batch(
            opts, self._c_training,
            self._real_points_ph, fake_images,
            self._is_training_ph, False)
        return res, res_fake

class MNISTLabelGan(ImageGan):
    """Architecture for MNIST from "Improved techniques for training GANs"

    """

    def generator(self, opts, noise, is_training, reuse=False):

        with tf.variable_scope("GENERATOR", reuse=reuse):

            h0 = ops.linear(opts, noise, 100, scope='h0_lin')
            h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1', scale=False)
            h0 = tf.nn.softplus(h0)
            h1 = ops.linear(opts, h0, 100, scope='h1_lin')
            h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2', scale=False)
            h1 = tf.nn.softplus(h1)
            h2 = ops.linear(opts, h1, 28 * 28, scope='h2_lin')
            # h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            h2 = tf.reshape(h2, [-1, 28, 28, 1])

        if opts['input_normalize_sym']:
            return tf.nn.tanh(h2)
        else:
            return tf.nn.sigmoid(h2)

    def discriminator(self, opts, input_, is_training,
                      prefix='DISCRIMINATOR', reuse=False):

        shape = tf.shape(input_)
        num = shape[0]

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = input_
            h0 = tf.add(h0, tf.random_normal(shape, stddev=0.3))
            h0 = ops.linear(opts, h0, 1000, scope='h0_linear')
            # h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
            h0 = tf.nn.relu(h0)
            h1 = tf.add(h0, tf.random_normal([num, 1000], stddev=0.5))
            h1 = ops.linear(opts, h1, 500, scope='h1_linear')
            # h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            h1 = tf.nn.relu(h1)
            h2 = tf.add(h1, tf.random_normal([num, 500], stddev=0.5))
            h2 = ops.linear(opts, h2, 250, scope='h2_linear')
            # h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            h2 = tf.nn.relu(h2)
            h3 = tf.add(h2, tf.random_normal([num, 250], stddev=0.5))
            h3 = ops.linear(opts, h3, 250, scope='h3_linear')
            # h3 = ops.batch_norm(opts, h3, is_training, reuse, scope='bn_layer4')
            h3 = tf.nn.relu(h3)
            h4 = tf.add(h3, tf.random_normal([num, 250], stddev=0.5))
            h4 = ops.linear(opts, h4, 250, scope='h4_linear')
            # h4 = ops.batch_norm(opts, h4, is_training, reuse, scope='bn_layer5')
            h4 = tf.nn.relu(h4)
            h5 = ops.linear(opts, h4, 10, scope='h5_linear')

        return h5, h3

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to GAN implementation.

        """
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        real_points_unl_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        is_training_ph = tf.placeholder(tf.bool, name='is_train_ph')
        dropout_rate_ph = tf.placeholder(tf.float32)
        # labels_ph = tf.placeholder(tf.int8, [None, 10])
        labels_ph = tf.placeholder(tf.int64, [None])
        lr_ph = tf.placeholder(tf.float32)


        # Operations
        G = self.generator(opts, noise_ph, is_training_ph)
        # We use conv2d_transpose in the generator, which results in the
        # output tensor of undefined shapes. However, we statically know
        # the shape of the generator output, which is [-1, dim1, dim2, dim3]
        # where (dim1, dim2, dim3) is given by self._data.data_shape
        # G.set_shape([None] + list(self._data.data_shape))

        # Here we follow a proposal of "Improved techniques for training
        # GANs" paper, Section 5

        d_logits_real, _ = self.discriminator(opts, real_points_ph, is_training_ph)
        d_logits_real_unl, d_features_real_unl = self.discriminator(
            opts, real_points_unl_ph, is_training_ph, reuse=True)
        d_logits_fake, d_features_fake = self.discriminator(
            opts, G, is_training_ph, reuse=True)

        d_loss_labelled = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=d_logits_real, labels=labels_ph))
        correct_predictions = tf.equal(
            tf.argmax(d_logits_real, axis=1),
            # tf.argmax(labels_ph, axis=1))
            labels_ph)
        d_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # 0 / 1 labels:
        # Z_real = ops.log_sum_exp(d_logits_real_unl)
        # Z_fake = ops.log_sum_exp(d_logits_fake)
        # D_real = Z_real - tf.nn.softplus(ops.log_sum_exp(d_logits_real_unl))
        # D_real = tf.Print(D_real, [D_real], 'Res:')

        # D_fake = -tf.nn.softplus(ops.log_sum_exp(d_logits_fake))
        # D_fake = tf.Print(D_fake, [D_fake])
        # d_loss_unl = - tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)

        # Label smoothing
        Z_real = ops.log_sum_exp(d_logits_real_unl)
        Z_fake = ops.log_sum_exp(d_logits_fake)
        cross_entropy_real_0 = Z_real - tf.nn.softplus(
            ops.log_sum_exp(d_logits_real_unl))
        # cross_entropy_real_0 = tf.Print(cross_entropy_real_0,
        #                               [tf.exp(cross_entropy_real_0)],
        #                               'D(X):')
        cross_entropy_real = 0.65 * cross_entropy_real_0 + 0.35 * (
            -tf.nn.softplus(ops.log_sum_exp(d_logits_real_unl)))
        cross_entropy_fake_0 = -tf.nn.softplus(
            ops.log_sum_exp(d_logits_fake))
        # cross_entropy_fake_0 = tf.Print(cross_entropy_fake_0,
        #                               [tf.exp(cross_entropy_fake_0)],
        #                               '1-D(G(Z)):')
        cross_entropy_fake = 1. * cross_entropy_fake_0 + 0. * (
            Z_fake - tf.nn.softplus(ops.log_sum_exp(d_logits_fake)))

        d_loss_unl = - tf.reduce_mean(cross_entropy_fake) \
            - tf.reduce_mean(cross_entropy_real)

        d_loss = d_loss_labelled + 0.5 * d_loss_unl

        # Log trick:
        # g_loss = -(ops.log_sum_exp(d_logits_fake) + cross_entropy_fake_0)
        # No log trick:
        # g_loss = tf.reduce_mean(cross_entropy_fake_0)
        # Feature matching
        f_mean_fake = tf.reduce_mean(d_features_fake, axis=0)
        f_mean_real = tf.reduce_mean(d_features_real_unl, axis=0)
        g_loss = tf.reduce_mean(tf.square(f_mean_fake - f_mean_real))

        c_logits_real, _ = self.discriminator(
            opts, real_points_ph, is_training_ph, prefix='CLASSIFIER')
        c_logits_fake, _ = self.discriminator(
            opts, fake_points_ph, is_training_ph,
            prefix='CLASSIFIER', reuse=True)
        c_training_logits, _ = self.discriminator(
            opts, real_points_ph, is_training_ph,
            prefix='CLASSIFIER', reuse=True)

        CZ_real = ops.log_sum_exp(c_logits_real)
        CD_real = CZ_real - tf.nn.softplus(ops.log_sum_exp(c_logits_real))
        CD_fake = -tf.nn.softplus(ops.log_sum_exp(c_logits_fake))
        c_loss = - tf.reduce_mean(CD_real) - tf.reduce_mean(CD_fake)

        c_training = tf.exp(CD_real)

        # d_optim_op = ops.optimizer(opts, 'd')
        # g_optim_op = ops.optimizer(opts, 'g')

        # def debug_grads(grad, var):
        #     _grad =  tf.Print(
        #         grad, # grads_and_vars,
        #         [tf.global_norm([grad])],
        #         'Global grad norm of %s: ' % var.name)
        #     return _grad, var

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        g_vars = [var for var in t_vars if 'GENERATOR/' in var.name]

        # d_grads_and_vars = [debug_grads(grad, var) for (grad, var) in \
        #     d_optim_op.compute_gradients(d_loss, var_list=d_vars)]
        # g_grads_and_vars = [debug_grads(grad, var) for (grad, var) in \
        #     g_optim_op.compute_gradients(g_loss, var_list=g_vars)]
        # d_optim = d_optim_op.apply_gradients(d_grads_and_vars)
        # g_optim = g_optim_op.apply_gradients(g_grads_and_vars)


        d_optim = tf.train.AdamOptimizer(lr_ph, beta1=opts["opt_beta1"])
        g_optim = tf.train.AdamOptimizer(lr_ph, beta1=opts["opt_beta1"])
        # g_optim = tf.train.GradientDescentOptimizer(lr_ph)
        d_optim = d_optim.minimize(d_loss, var_list=d_vars)
        g_optim = g_optim.minimize(g_loss, var_list=g_vars)

        c_vars = [var for var in t_vars if 'CLASSIFIER/' in var.name]
        c_optim = ops.optimizer(opts).minimize(c_loss, var_list=c_vars)

        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph
        self._real_points_unl_ph = real_points_unl_ph
        self._is_training_ph = is_training_ph
        self._dropout_rate_ph = dropout_rate_ph
        self._G = G
        self._d_loss = d_loss
        self._g_loss = g_loss
        self._c_loss = c_loss
        self._c_training = c_training
        self._g_optim = g_optim
        self._d_optim = d_optim
        self._c_optim = c_optim
        self._labels_ph = labels_ph
        self._d_accuracy = d_accuracy
        self._g_loss = g_loss
        self._lr_ph = lr_ph

        logging.debug("Building Graph Done.")

    def _train_internal(self, opts):
        """Train a GAN model.

        """

        train_data = self._data.data[:60000]
        train_labels = self._data.labels[:60000]
        train_weights = self._data_weights[:60000]
        train_weights = train_weights / np.sum(train_weights)
        test_data = self._data.data[60000:]
        test_labels = self._data.labels[60000:]
        batches_num = len(train_data) / opts['batch_size']
        train_size = len(train_data)

        counter = 0
        logging.debug('Training GAN')
        lr_g = opts['opt_g_learning_rate']
        lr_d = opts['opt_d_learning_rate']
        accuracy = 0.
        for _epoch in xrange(opts["gan_epoch_num"]):
            for _idx in xrange(batches_num):
                # logging.debug('Step %d of %d' % (_idx, batches_num ) )
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=train_weights)
                data_ids_unl = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=train_weights)
                batch_images = train_data[data_ids].astype(np.float)
                batch_images_unl = train_data[data_ids_unl].astype(np.float)
                batch_noise = utils.generate_noise(opts, opts['batch_size'])
                # Update discriminator parameters
                # labels_oh = utils.one_hot(self._data.labels[data_ids])
                labels_oh = train_labels[data_ids]
                lr = lr_d * min(1., 1. - ((0. + _epoch) / opts['gan_epoch_num']))
                for _iter in xrange(opts['d_steps']):
                    _ = self._session.run(
                        self._d_optim,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._real_points_unl_ph: batch_images_unl,
                                   self._is_training_ph: True,
                                   self._lr_ph: lr,
                                   self._noise_ph: batch_noise,
                                   self._labels_ph: labels_oh})
                # Update generator parameters
                lr = lr_g * min(1., 1. - ((0. + _epoch) / opts['gan_epoch_num']))
                for _iter in xrange(opts['g_steps']):
                    _ = self._session.run(
                        self._g_optim,
                        feed_dict={self._noise_ph: batch_noise,
                                   self._is_training_ph: True,
                                   self._lr_ph: lr,
                                   self._real_points_unl_ph: batch_images_unl})
                counter += 1

                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    accuracy = self._d_accuracy.eval(
                        feed_dict={self._real_points_ph: test_data,
                                   self._is_training_ph: False,
                                   # self._labels_ph: utils.one_hot(self._data.labels[:1000])})
                                   self._labels_ph: test_labels})
                    g_loss = self._g_loss.eval(
                        feed_dict={self._noise_ph: batch_noise,
                                   self._is_training_ph: False,
                                   self._real_points_unl_ph: batch_images_unl})
                    logging.debug(
                        'Epoch:%3d/%d, batch:%4d/%d, lr_g=%.4f, D accuracy in telling digits:%f, G feature matching loss:%f' % \
                        (_epoch+1, opts['gan_epoch_num'], _idx+1, batches_num, lr, accuracy, g_loss))
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._G, self._noise_ph,
                        self._noise_for_plots[0:320],
                        self._is_training_ph, False)
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        points_to_plot,
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))
                if opts['early_stop'] > 0 and counter > opts['early_stop']:
                    break

    def _train_mixture_discriminator_internal(self, opts, fake_images):
        """Train a classifier separating true data from points in fake_images.

        """

        batches_num = self._data.num_points / opts['batch_size']
        logging.debug('Training a mixture discriminator')
        logging.debug('Using %d real points and %d fake ones' %\
                      (self._data.num_points, len(fake_images)))
        for epoch in xrange(opts["mixture_c_epoch_num"]):
            for idx in xrange(batches_num):
                ids = np.random.choice(len(fake_images), opts['batch_size'],
                                       replace=False)
                batch_fake_images = fake_images[ids]
                ids = np.random.choice(self._data.num_points, opts['batch_size'],
                                       replace=False)
                batch_real_images = self._data.data[ids]
                _ = self._session.run(
                    self._c_optim,
                    feed_dict={self._real_points_ph: batch_real_images,
                               self._fake_points_ph: batch_fake_images,
                               self._is_training_ph: True})

        # Evaluating trained classifier on real points
        res = self._run_batch(
            opts, self._c_training,
            self._real_points_ph, self._data.data,
            self._is_training_ph, False)

        # Evaluating trained classifier on fake points
        res_fake = self._run_batch(
            opts, self._c_training,
            self._real_points_ph, fake_images,
            self._is_training_ph, False)

        return res, res_fake

class BigImageGan(ImageGan):
    """A bit more flexible generator, compared to ImageGan.

    """

    def generator(self, opts, noise, is_training, reuse=False):
        """Generator function, suitable for bigger simple pictures.

        Args:
            noise: [num_points, dim] array, where dim is dimensionality of the
                latent noise space.
            is_training: bool, defines whether to use batch_norm in the train
                or test mode.
        Returns:
            [num_points, dim1, dim2, dim3] array, where the first coordinate
            indexes the points, which all are of the shape (dim1, dim2, dim3).
        """

        output_shape = self._data.data_shape # (dim1, dim2, dim3)
        # Computing the number of noise vectors on-the-go
        dim1 = tf.shape(noise)[0]
        num_filters = opts['g_num_filters']

        with tf.variable_scope("GENERATOR", reuse=reuse):

            height = output_shape[0] / 16
            width = output_shape[1] / 16
            h0 = ops.linear(opts, noise, num_filters * height * width,
                            scope='h0_lin')
            h0 = tf.reshape(h0, [-1, height, width, num_filters])
            h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
            h0 = tf.nn.relu(h0)
            _out_shape = [dim1, height * 2, width * 2, num_filters / 2]
            # for 128 x 128 does 8 x 8 --> 16 x 16
            h1 = ops.deconv2d(opts, h0, _out_shape, scope='h1_deconv')
            h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            h1 = tf.nn.relu(h1)
            _out_shape = [dim1, height * 4, width * 4, num_filters / 4]
            # for 128 x 128 does 16 x 16 --> 32 x 32 
            h2 = ops.deconv2d(opts, h1, _out_shape, scope='h2_deconv')
            h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            h2 = tf.nn.relu(h2)
            _out_shape = [dim1, height * 8, width * 8, num_filters / 8]
            # for 128 x 128 does 32 x 32 --> 64 x 64 
            h3 = ops.deconv2d(opts, h2, _out_shape, scope='h3_deconv')
            h3 = ops.batch_norm(opts, h3, is_training, reuse, scope='bn_layer4')
            h3 = tf.nn.relu(h3)
            _out_shape = [dim1, height * 16, width * 16, num_filters / 16]
            # for 128 x 128 does 64 x 64 --> 128 x 128 
            h4 = ops.deconv2d(opts, h3, _out_shape, scope='h4_deconv')
            h4 = ops.batch_norm(opts, h4, is_training, reuse, scope='bn_layer5')
            h4 = tf.nn.relu(h4)
            _out_shape = [dim1] + list(output_shape)
            # data_shape[0] x data_shape[1] x ? -> data_shape
            h5 = ops.deconv2d(opts, h4, _out_shape,
                              d_h=1, d_w=1, scope='h5_deconv')
            h5 = ops.batch_norm(opts, h5, is_training, reuse, scope='bn_layer6')

        if opts['input_normalize_sym']:
            return tf.nn.tanh(h5)
        else:
            return tf.nn.sigmoid(h5)


class ImageUnrolledGan(ImageGan):
    """A simple GAN implementation, suitable for pictures.

    """

    def __init__(self, opts, data, weights):

        # Losses of the copied discriminator network
        self._d_loss_cp = None
        self._d_optim_cp = None
        # Rolling back ops (assign variable values fo true
        # to copied discriminator network)
        self._roll_back = None

        ImageGan.__init__(self, opts, data, weights)

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to GAN implementation.

        """
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        is_training_ph = tf.placeholder(tf.bool, name='is_train_ph')

        # Operations
        G = self.generator(opts, noise_ph, is_training_ph)
        # We use conv2d_transpose in the generator, which results in the
        # output tensor of undefined shapes. However, we statically know
        # the shape of the generator output, which is [-1, dim1, dim2, dim3]
        # where (dim1, dim2, dim3) is given by self._data.data_shape
        G.set_shape([None] + list(self._data.data_shape))

        d_logits_real = self.discriminator(opts, real_points_ph, is_training_ph)
        d_logits_fake = self.discriminator(opts, G, is_training_ph, reuse=True)

        # Disccriminator copy for the unrolling steps
        d_logits_real_cp = self.discriminator(
            opts, real_points_ph, is_training_ph, prefix='DISCRIMINATOR_CP')
        d_logits_fake_cp = self.discriminator(
            opts, G, is_training_ph, prefix='DISCRIMINATOR_CP', reuse=True)


        c_logits_real = self.discriminator(
            opts, real_points_ph, is_training_ph, prefix='CLASSIFIER')
        c_logits_fake = self.discriminator(
            opts, fake_points_ph, is_training_ph, prefix='CLASSIFIER', reuse=True)
        c_training = tf.nn.sigmoid(
            self.discriminator(opts, real_points_ph, is_training_ph,
                               prefix='CLASSIFIER', reuse=True))

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        d_loss_real_cp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_real_cp, labels=tf.ones_like(d_logits_real_cp)))
        d_loss_fake_cp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake_cp,
                labels=tf.zeros_like(d_logits_fake_cp)))
        d_loss_cp = d_loss_real_cp + d_loss_fake_cp

        if opts['objective'] == 'JS':
            g_loss = - d_loss_cp
        elif opts['objective'] == 'JS_modified':
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logits_fake_cp,
                    labels=tf.ones_like(d_logits_fake_cp)))
        else:
            assert False, 'No objective %r implemented' % opts['objective']

        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_real, labels=tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
               logits=c_logits_fake, labels=tf.zeros_like(c_logits_fake)))
        c_loss = c_loss_real + c_loss_fake

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'DISCRIMINATOR/' in var.name]
        d_vars_cp = [var for var in t_vars if 'DISCRIMINATOR_CP/' in var.name]
        c_vars = [var for var in t_vars if 'CLASSIFIER/' in var.name]
        g_vars = [var for var in t_vars if 'GENERATOR/' in var.name]

        # Ops to roll back the variable values of discriminator_cp
        # Will be executed each time before the unrolling steps
        with tf.variable_scope('assign'):
            roll_back = []
            for var, var_cp in zip(d_vars, d_vars_cp):
                roll_back.append(tf.assign(var_cp, var))

        d_optim = ops.optimizer(opts, 'd').minimize(d_loss, var_list=d_vars)
        d_optim_cp = ops.optimizer(opts, 'd').minimize(
           d_loss_cp, var_list=d_vars_cp)
        c_optim = ops.optimizer(opts).minimize(c_loss, var_list=c_vars)
        g_optim = ops.optimizer(opts, 'g').minimize(g_loss, var_list=g_vars)

        # writer = tf.summary.FileWriter(opts['work_dir']+'/tensorboard', self._session.graph)

        # d_optim_op = ops.optimizer(opts, 'd')
        # g_optim_op = ops.optimizer(opts, 'g')

        # def debug_grads(grad, var):
        #     _grad =  tf.Print(
        #         grad, # grads_and_vars,
        #         [tf.global_norm([grad])], 
        #         'Global grad norm of %s: ' % var.name)
        #     return _grad, var

        # d_grads_and_vars = [debug_grads(grad, var) for (grad, var) in \
        #     d_optim_op.compute_gradients(d_loss, var_list=d_vars)]
        # g_grads_and_vars = [debug_grads(grad, var) for (grad, var) in \
        #     g_optim_op.compute_gradients(g_loss, var_list=g_vars)]
        # d_optim = d_optim_op.apply_gradients(d_grads_and_vars)
        # g_optim = g_optim_op.apply_gradients(g_grads_and_vars)

        c_vars = [var for var in t_vars if 'CLASSIFIER/' in var.name]
        c_optim = ops.optimizer(opts).minimize(c_loss, var_list=c_vars)

        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph
        self._is_training_ph = is_training_ph
        self._G = G
        self._roll_back = roll_back
        self._d_loss = d_loss
        self._d_loss_cp = d_loss_cp
        self._g_loss = g_loss
        self._c_loss = c_loss
        self._c_training = c_training
        self._g_optim = g_optim
        self._d_optim = d_optim
        self._d_optim_cp = d_optim_cp
        self._c_optim = c_optim

        logging.debug("Building Graph Done.")


    def _train_internal(self, opts):
        """Train a GAN model.

        """
        batches_num = self._data.num_points / opts['batch_size']
        train_size = self._data.num_points

        counter = 0
        logging.debug('Training GAN')
        for _epoch in xrange(opts["gan_epoch_num"]):
            for _idx in TQDM(opts, xrange(batches_num),
                             desc='Epoch %2d/%2d' %\
                             (_epoch + 1, opts["gan_epoch_num"])):
                # logging.debug('Step %d of %d' % (_idx, batches_num ) )
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                            replace=False, p=self._data_weights)
                batch_images = self._data.data[data_ids].astype(np.float)
                batch_noise = utils.generate_noise(opts, opts['batch_size'])
                # Update discriminator parameters
                for _iter in xrange(opts['d_steps']):
                    _ = self._session.run(
                        self._d_optim,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                # Roll back discriminator_cp's variables
                self._session.run(self._roll_back)
                # Unrolling steps
                for _iter in xrange(opts['unrolling_steps']):
                    self._session.run(
                        self._d_optim_cp,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                # Update generator parameters
                for _iter in xrange(opts['g_steps']):
                    _ = self._session.run(
                        self._g_optim,
                        feed_dict={self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                counter += 1

                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    logging.debug(
                        'Epoch: %d/%d, batch:%d/%d' % \
                        (_epoch+1, opts['gan_epoch_num'], _idx+1, batches_num))
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._G, self._noise_ph,
                        self._noise_for_plots[0:320],
                        self._is_training_ph, False)
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        points_to_plot,
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))
                if opts['early_stop'] > 0 and counter > opts['early_stop']:
                    break
