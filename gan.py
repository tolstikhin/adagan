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
        self._data_weights = weights
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

        # Variables
        self._inv_input = None

        # Optimizers
        self._g_optim = None
        self._d_optim = None
        self._c_optim = None
        self._inv_optim = None

        with self._session.as_default(), self._session.graph.as_default():
            logging.debug('Building the graph...')
            self._build_model_internal(opts)
            if opts['inverse_metric']:
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

    def invert_point(self, opts, image):
        """Invert the learned generator function for image.

        Args:
            image: numpy array of shape data_shape.

        """
        assert self._trained, 'Can not invert, not trained yet.'
        data_shape = self._data.data_shape
        with self._session.as_default(), self._session.graph.as_default():
            target_ph = self._inv_target_ph
            params = self._inv_input
            loss = self._inv_loss
            optim = self._inv_optim
            opt_vals = []
            opt_params = []
            for _start in xrange(5):
                all_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="inversion")
                self._session.run(tf.variables_initializer(all_vars))
                print 'Loss = ', loss.eval(feed_dict={target_ph:image})
                prev_val = 1e4
                check_every = 50
                steps = 1
                while True:
                    self._session.run(
                        optim, feed_dict={target_ph:image})
                    if steps % check_every == 0:
                        err = loss.eval(feed_dict={target_ph:image})
                        logging.debug('Init %02d, steps %d, loss %f' %\
                                      (_start, steps, err))
                        relative_improvement = np.abs(prev_val - err) / prev_val
                        if relative_improvement < 1e-4:
                            opt_vals.append(err)
                            opt_params.append(self._session.run(params))
                            break
                        prev_val = err
                    steps += 1
            best_id = sorted(zip(opt_vals, range(len(opt_vals))))[0][1]
            opt_val = opt_vals[best_id]
            opt_param = opt_params[best_id]
            opt_image = self._G.eval(
                feed_dict={self._noise_ph:opt_param,
                           self._is_training_ph:False})[0]

            return opt_image, opt_params[best_id], opt_vals[best_id]

    def _add_inversion_ops(self, opts):
        data_shape = self._data.data_shape
        with tf.variable_scope("inversion"):
            target_ph = tf.placeholder(
                tf.float32, data_shape, name='target_ph')
            params = tf.get_variable(
                "inverted", [1, opts['latent_space_dim']],
                tf.float32, tf.random_normal_initializer(stddev=1.))
        reconstructed_image = self.generator(
            opts, params, is_training=False, reuse=True)
        with tf.variable_scope("inversion"):
            loss = tf.reduce_mean(
                tf.square(tf.sub(reconstructed_image, target_ph)))
            optim = tf.train.MomentumOptimizer(0.1, 0.9)
            optim = optim.minimize(loss, var_list=[params])
        self._inv_target_ph = target_ph
        self._inv_input = params
        self._inv_optim = optim
        self._inv_loss = loss

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
            h0 = ops.linear(opts, noise, 10, 'h0_lin')
            h0 = tf.nn.relu(h0)
            h1 = ops.linear(opts, h0, 5, 'h1_lin')
            h1 = tf.nn.relu(h1)
            h2 = ops.linear(opts, h1, np.prod(output_shape), 'h2_lin')
            h2 = tf.reshape(h2, [-1] + list(output_shape))

        return h2

    def discriminator(self, opts, input_,
                      prefix='DISCRIMINATOR', reuse=False):
        """Discriminator function, suitable for simple toy experiments.

        """
        shape = input_.get_shape().as_list()
        assert len(shape) > 0, 'No inputs to discriminate.'

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.linear(opts, input_, 50, 'h0_lin')
            h0 = tf.nn.relu(h0)
            h1 = ops.linear(opts, h0, 30, 'h1_lin')
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
                d_logits_real, tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_fake, tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_fake, tf.ones_like(d_logits_fake)))

        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                c_logits_real, tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                c_logits_fake, tf.zeros_like(c_logits_fake)))
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
            for _idx in TQDM(opts, xrange(batches_num),
                             desc='Epoch %2d/%2d'% (_epoch+1,opts["gan_epoch_num"])):
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
                if opts['verbose'] and counter % 100 == 0:
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._G, self._noise_ph,
                        self._noise_for_plots[0:300])
                    metrics.make_plots(
                        opts,
                        counter,
                        self._data.data[0:300],
                        points_to_plot,
                        prefix='gan_e%d_mb%d_' % (_epoch, _idx))



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
            h0 = ops.linear(opts, noise, 128, 'h0_lin')
            h0 = tf.nn.tanh(h0)
            h1 = ops.linear(opts, h0, 128, 'h1_lin')
            h1 = tf.nn.tanh(h1)
            h2 = ops.linear(opts, h1, np.prod(output_shape), 'h2_lin')
            h2 = tf.reshape(h2, [-1] + list(output_shape))

        return h2

    def discriminator(self, opts, input_,
                      prefix='DISCRIMINATOR', reuse=False):
        """Discriminator function, suitable for simple toy experiments.

        """
        shape = input_.get_shape().as_list()
        assert len(shape) > 0, 'No inputs to discriminate.'

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.linear(opts, input_, 128, 'h0_lin')
            h0 = tf.nn.tanh(h0)
            h1 = ops.linear(opts, h0, 128, 'h1_lin')
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
                d_logits_real, tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_fake, tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        d_loss_real_cp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_real_cp, tf.ones_like(d_logits_real_cp)))
        d_loss_fake_cp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_fake_cp, tf.zeros_like(d_logits_fake_cp)))
        d_loss_cp = d_loss_real_cp + d_loss_fake_cp

        if opts['objective'] == 'JS':
            g_loss = - d_loss_cp
        elif opts['objective'] == 'JS_modified':
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    d_logits_fake_cp, tf.ones_like(d_logits_fake_cp)))
        else:
            assert False, 'No objective %r implemented' % opts['objective']

        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                c_logits_real, tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                c_logits_fake, tf.zeros_like(c_logits_fake)))
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
                             desc='Epoch %2d/%2d'% (_epoch+1,opts["gan_epoch_num"])):
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
                if opts['verbose'] and counter % 100 == 0:
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._G, self._noise_ph,
                        self._noise_for_plots[0:300])
                    metrics.make_plots(
                        opts,
                        counter,
                        self._data.data[0:300],
                        points_to_plot,
                        prefix='gan_e%d_mb%d_' % (_epoch, _idx))



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
        shape = input_.get_shape().as_list()
        assert len(shape) > 0, 'No inputs to discriminate.'
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
                d_logits_real, tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_fake, tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_fake, tf.ones_like(d_logits_fake)))

        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                c_logits_real, tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                c_logits_fake, tf.zeros_like(c_logits_fake)))
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
        #         [tf.global_norm([grad])], # tf.global_norm([grad for (grad, var) in grads_and_vars]).get_shape(),
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
                        self._noise_for_plots[0:3 * 16],
                        self._is_training_ph, False)
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        points_to_plot,
                        prefix='sample_e%02d_mb%05d_' % (_epoch, _idx))
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


class ImageUnrolledGan(ImageGan):
    """A simple GAN implementation, suitable for pictures.

    """

    def __init__(self, opts, data, weights):

        # One more placeholder for batch norm
        self._is_training_ph = None
        # Losses of the copied discriminator network 
        self._d_loss_cp = None
        self._d_optim_cp = None
        # Rolling back ops (assign variable values fo true
        # to copied discriminator network)
        self._roll_back = None

        Gan.__init__(self, opts, data, weights)

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
                d_logits_real, tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_fake, tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        d_loss_real_cp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_real_cp, tf.ones_like(d_logits_real_cp)))
        d_loss_fake_cp = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                d_logits_fake_cp, tf.zeros_like(d_logits_fake_cp)))
        d_loss_cp = d_loss_real_cp + d_loss_fake_cp

        if opts['objective'] == 'JS':
            g_loss = - d_loss_cp
        elif opts['objective'] == 'JS_modified':
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    d_logits_fake_cp, tf.ones_like(d_logits_fake_cp)))
        else:
            assert False, 'No objective %r implemented' % opts['objective']

        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                c_logits_real, tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                c_logits_fake, tf.zeros_like(c_logits_fake)))
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

        # d_optim_op = ops.optimizer(opts, 'd')
        # g_optim_op = ops.optimizer(opts, 'g')

        # def debug_grads(grad, var):
        #     _grad =  tf.Print(
        #         grad, # grads_and_vars,
        #         [tf.global_norm([grad])], # tf.global_norm([grad for (grad, var) in grads_and_vars]).get_shape(),
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
                             desc='Epoch %2d/%2d'% (_epoch+1,opts["gan_epoch_num"])):
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
                    # logging.debug(
                    #     'Epoch: %d/%d, batch:%d/%d' % \
                    #     (_epoch+1, opts['gan_epoch_num'], _idx+1, batches_num))
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._G, self._noise_ph,
                        self._noise_for_plots[0:16],
                        self._is_training_ph, False)
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        points_to_plot,
                        prefix='sample_e%02d_mb%05d_' % (_epoch, _idx))
                if opts['early_stop'] > 0 and counter > opts['early_stop']:
                    break
