# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""The class implementing AdaGAN iterative training procedure.

"""

import logging
import numpy as np
import gan as GAN
import vae as VAE
import pot as POT
from utils import ArraySaver
from metrics import Metrics
import utils

class AdaGan(object):
    """This class implements the AdaGAN meta-algorithm.

    The class provides the 'make_step' method, which calls Gan.train()
    method to train the next Generator function. It also updates the
    weights of training points and takes care of mixture weights for
    newly trained mixture components.

    The same class can be used to implement the bagging, i.e. uniform
    mixture of independently trained GANs. This is controlled by
    opts['is_bagging'].
    """

    # pylint: disable=too-many-instance-attributes
    # We need this many.
    def __init__(self, opts, data):
        self.steps_total = opts['adagan_steps_total']
        self.steps_made = 0
        num = data.num_points
        self._data_num = num
        self._data_weights = np.ones(num) / (num + 0.)
        self._mixture_weights = np.zeros(0)
        self._beta_heur = opts['beta_heur']
        self._saver = ArraySaver('disk', workdir=opts['work_dir'])
        # Which GAN architecture should we use?
        pic_datasets = ['mnist',
                        'mnist3',
                        'guitars',
                        'cifar10']
        supervised_pic_datasets = ['mnist',
                                   'mnist3',
                                   'cifar10']
        gan_class = None
        if opts['dataset'] in ('gmm', 'circle_gmm'):
            if opts['unrolled'] is True:
                gan_class = GAN.ToyUnrolledGan
            else:
                gan_class = GAN.ToyGan
        elif opts['dataset'] in pic_datasets:
            if opts['unrolled']:
                gan_class = GAN.ImageUnrolledGan
                # gan_class = GAN.ToyUnrolledGan
            else:
                if opts['vae']:
                    gan_class = VAE.ImageVae
                    assert opts['latent_space_distr'] == 'normal',\
                        'VAE works only with Gaussian prior'
                elif opts['pot']:
                    gan_class = POT.ImagePot
                else:
                    gan_class = GAN.ImageGan
                    if opts['dataset'] in supervised_pic_datasets\
                            and opts['conditional']:
                        gan_class = GAN.MNISTLabelGan
        elif opts['dataset'] == 'guitars':
            if opts['unrolled']:
                gan_class = GAN.ImageUnrolledGan
            else:
                gan_class = GAN.BigImageGan
        else:
            assert False, "We don't have any other GAN implementations yet..."
        self._gan_class = gan_class
        if opts["inverse_metric"]:
            inv_num = opts['inverse_num']
            assert inv_num < data.num_points, \
                'Number of points to invert larger than a training set'
            inv_num = min(inv_num, data.num_points)
            self._invert_point_ids = np.random.choice(
                data.num_points, inv_num, replace=False)
            self._invert_losses = np.zeros((self.steps_total, inv_num))

    def make_step(self, opts, data):
        """Makes one AdaGAN step and takes care of all necessary updates.

        This function runs an individual instance of GAN on a reweighted
        dataset. Before doing so, it first computes the mixture weight of
        the next component generator and updates the weights of data points.
        Finally, it saves the sample from the newly created generator for
        future use.

        Args:
            opts: A dict of options.
            data: An instance of DataHandler. Contains the training set and all
                the relevant info about it.
        """

        with self._gan_class(opts, data, self._data_weights) as gan:

            beta = self._next_mixture_weight(opts)
            if self.steps_made > 0 and not opts['is_bagging']:
                # We first need to update importance weights
                # Two cases when we don't need to do this are:
                # (a) We are running the very first GAN instance
                # (b) We are bagging, in which case the weughts are always uniform
                self._update_data_weights(opts, gan, beta, data)
                gan._data_weights = np.copy(self._data_weights)

            # Train GAN
            gan.train(opts)
            # Save a sample
            logging.debug('Saving a sample from the trained component...')
            sample = gan.sample(opts, opts['samples_per_component'])
            self._saver.save('samples{:02d}.npy'.format(self.steps_made), sample)
            metrics = Metrics()
            metrics.make_plots(opts, self.steps_made, data.data,
                               sample[:min(len(sample), 320)],
                               prefix='component_')
            #3. Invert the generator, while we still have the graph alive.
            if opts["inverse_metric"]:
                logging.debug('Inverting data points...')
                ids = self._invert_point_ids
                images_hat, z, err_per_point, norms = gan.invert_points(
                    opts, data.data[ids])
                plot_pics = []
                for _id in xrange(min(16 * 8, len(ids))):
                    plot_pics.append(images_hat[_id])
                    plot_pics.append(data.data[ids[_id]])
                metrics.make_plots(
                    opts, self.steps_made, data.data,
                    np.array(plot_pics),
                    prefix='inverted_')
                logging.debug('Inverted with mse=%.5f, std=%.5f' %\
                        (np.mean(err_per_point), np.std(err_per_point)))
                self._invert_losses[self.steps_made] = err_per_point
                self._saver.save(
                    'mse{:02d}.npy'.format(self.steps_made), err_per_point)
                self._saver.save(
                    'mse_norms{:02d}.npy'.format(self.steps_made), norms)
                logging.debug('Inverting done.')

        if self.steps_made == 0:
            self._mixture_weights = np.array([beta])
        else:
            scaled_old_weights = [v * (1.0 - beta) for v in self._mixture_weights]
            self._mixture_weights = np.array(scaled_old_weights + [beta])
        self.steps_made += 1

    def sample_mixture(self, num=100):
        """Sample num elements from the current AdaGAN mixture of generators.

        In this code we are not storing individual TensorFlow graphs
        corresponding to every one of the already trained component generators.
        Instead, we sample enough of points once per every trained
        generator and store these samples. Later, in order to sample from the
        mixture, we first define which component to sample from and then
        pick points uniformly from the corresponding stored sample.

        """

        #First we define how many points do we need
        #from each of the components
        component_ids = []
        for _ in xrange(num):
            new_id = np.random.choice(self.steps_made, 1,
                                      p=self._mixture_weights)[0]
            component_ids.append(new_id)
        points_per_component = [component_ids.count(i)
                                for i in xrange(self.steps_made)]

        # Next we sample required number of points per component
        sample = []
        for comp_id  in xrange(self.steps_made):
            _num = points_per_component[comp_id]
            if _num == 0:
                continue
            comp_samples = self._saver.load('samples{:02d}.npy'.format(comp_id))
            for _ in xrange(_num):
                sample.append(
                    comp_samples[np.random.randint(len(comp_samples))])

        # Finally we shuffle
        res = np.array(sample)
        np.random.shuffle(res)

        return res


    def _next_mixture_weight(self, opts):
        """Returns a weight, corresponding to the next mixture component.

        """
        if self.steps_made == 0:
            return 1.
        else:
            if self._beta_heur == 'uniform' or opts['is_bagging']:
                # This weighting scheme will correspond to the uniform mixture
                # of the resulting component generators. Thus this scheme can
                # be also used for bagging.
                return 1./(self.steps_made + 1.)
            elif self._beta_heur == 'constant':
                assert opts["beta_constant"] >= 0.0, 'Beta should be nonnegative'
                assert opts["beta_constant"] <= 1.0, 'Beta should be < 1'
                return opts["beta_constant"]
            else:
                assert False, 'Unknown beta heuristic'

    def _update_data_weights(self, opts, gan, beta, data):
        """Update the weights of data points based on the current mixture.

        This function defines a discrete distribution over the training points
        which will be used by GAN while sampling mini batches. For AdaGAN
        algorithm we have several heuristics, including the one based on
        the theory provided in 'AdaGAN: Boosting Generative Models'.
        """
        # 1. First we need to train the big classifier, separating true data
        # from the fake one sampled from the current mixture generator.
        # Its outputs are already normalized in [0,1] with sigmoid
        prob_real_data = self._get_prob_real_data(opts, gan, data)
        prob_real_data = prob_real_data.flatten()
        density_ratios = (1. - prob_real_data) / (prob_real_data + 1e-8)
        self._data_weights = self._compute_data_weights(opts,
                                                        density_ratios, beta)
        # We may also print some debug info on the computed weights
        utils.debug_updated_weights(opts, self.steps_made,
                                    self._data_weights, data)


    def _compute_data_weights(self, opts, density_ratios, beta):
        """Compute a discrite distribution over the training points.

        Given per-point estimates of dP_current_model(x)/dP_data(x), compute
        the discrite distribution over the training points, which is called
        W_t in the arXiv paper, see Algorithm 1.
        """

        heur = opts['weights_heur']
        if heur == 'topk':
            return self._compute_data_weights_topk(opts, density_ratios)
        elif heur == 'theory_star':
            return self._compute_data_weights_theory_star(beta, density_ratios)
        elif heur == 'theory_dagger':
            return self._compute_data_weights_theory_dagger(beta, density_ratios)
        else:
            assert False, 'Unknown weights heuristic'


    def _compute_data_weights_topk(self, opts, density_ratios):
        """Put a uniform distribution on K points with largest prob real data.

        This is a naiive heuristic which makes next GAN concentrate on those
        points of the training set, which were classified correctly with
        largest margins. I.e., out current mixture model is not capable of
        generating points looking similar to these ones.
        """
        threshold = np.percentile(density_ratios,
                                  opts["topk_constant"]*100.0)
        # Note that largest prob_real_data corresponds to smallest density
        # ratios.
        mask = density_ratios <= threshold
        data_weights = np.zeros(self._data_num)
        data_weights[mask] = 1.0 / np.sum(mask)
        return data_weights

    def _compute_data_weights_theory_star(self, beta, ratios):
        """Theory-inspired reweighting of training points.

        Refer to Section 3.1 of the arxiv paper
        """
        num = self._data_num
        ratios_sorted = np.sort(ratios)
        cumsum_ratios = np.cumsum(ratios_sorted)
        is_found = False
        # We first find the optimal lambda* which is guaranteed to exits.
        # While Lemma 5 guarantees that lambda* <= 1, in practice this may
        # not be the case, as we replace dPmodel/dPdata by (1-D)/D.
        for i in xrange(num):
            # Computing lambda from equation (18) of the arxiv paper
            _lambda = beta * num * (1. + (1.-beta) / beta \
                    / num * cumsum_ratios[i]) / (i + 1.)
            if i == num - 1:
                if _lambda >= (1. - beta) * ratios_sorted[-1]:
                    is_found = True
                    break
            else:
                if _lambda <= (1 - beta) * ratios_sorted[i + 1] \
                        and _lambda >= (1 - beta) * ratios_sorted[i]:
                    is_found = True
                    break
        # Next we compute the actual weights using equation (17)
        data_weights = np.zeros(num)
        if is_found:
            _lambdamask = ratios <= (_lambda / (1.-beta))
            data_weights[_lambdamask] = (_lambda -
                                         (1-beta)*ratios[_lambdamask]) / num / beta
            logging.debug(
                'Lambda={}, sum={}, deleted points={}'.format(
                    _lambda,
                    np.sum(data_weights),
                    1.0 * (num - sum(_lambdamask)) / num))
            # This is a delicate moment. Ratios are supposed to be
            # dPmodel/dPdata. However, we are using a heuristic
            # esplained around (16) in the arXiv paper. So the
            # resulting weights do not necessarily need to some
            # to one.
            data_weights = data_weights / np.sum(data_weights)
            return data_weights
        else:
            logging.debug(
                '[WARNING] Lambda search failed, passing uniform weights')
            data_weights = np.ones(num) / (num + 0.)
            return data_weights

    def _compute_data_weights_theory_dagger(self, beta, ratios):
        """Theory-inspired reweighting of training points.

        Refer to Theorem 2 of the arxiv paper
        """
        num = self._data_num
        ratios_sorted = np.sort(ratios)
        cumsum_ratios = np.cumsum(ratios_sorted)
        is_found = False
        # We first find the optimal lambda* which is guaranteed to exits.
        for i in range(int(np.floor(num * beta - 1)), num):
            # Computing lambda
            if (i + 1.) / num < beta:
                continue
            _lambda = ((i + 1.) / num - beta) / (1. - beta) * num \
                / (cumsum_ratios[i] + 1e-7)
            if i == num - 1:
                if _lambda < 1. / (1. - beta) / (ratios_sorted[i] + 1e-7):
                    is_found = True
                    break
            else:
                if _lambda < 1. / (1. - beta) / (ratios_sorted[i] + 1e-7) \
                        and _lambda >= 1. / (1. - beta) / \
                            (ratios_sorted[i + 1] + 1e-7):
                    is_found = True
                    break
        # Next we compute the actual weights using equation (17)
        data_weights = np.zeros(num)
        if is_found:
            _lambdamask = ratios <= (1. / (1.-beta) / _lambda)
            data_weights[_lambdamask] = \
                (1. - _lambda * (1-beta) * ratios[_lambdamask]) / num / beta
            logging.debug(
                'Lambda={}, sum={}, deleted points={}'.format(
                    _lambda,
                    np.sum(data_weights),
                    1.0 * (num - sum(_lambdamask)) / num))
            # This is a delicate moment. Ratios are supposed to be
            # dPmodel/dPdata. However, we are using a heuristic
            # esplained around (16) in the arXiv paper. So the
            # resulting weights do not necessarily need to some
            # to one.
            data_weights = data_weights / np.sum(data_weights)
            return data_weights
        else:
            logging.warning(
                '[WARNING] Lambda search failed, passing uniform weights')
            data_weights = np.ones(num) / (num + 0.)
            return data_weights

    def _get_prob_real_data(self, opts, gan, data):
        """Train a classifier, separating true data from the current mixture.

        Returns:
            (data.num_points,) NumPy array, containing probabilities of true
            data. I.e., output of the sigmoid function.
        """
        num_fake_images = data.num_points
        fake_images = self.sample_mixture(num_fake_images)
        prob_real, prob_fake = \
            gan.train_mixture_discriminator(opts, fake_images)
        # We may also plot fake / real points correctly/incorrectly classified
        # by the trained classifier just for debugging purposes
        if prob_fake is not None:
            utils.debug_mixture_classifier(opts, self.steps_made, prob_fake,
                                           fake_images, real=False)
        utils.debug_mixture_classifier(opts, self.steps_made, prob_real,
                                       data.data, real=True)
        return prob_real
