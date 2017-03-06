# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""Training AdaGAN on various datasets.

Refer to the arXiv paper 'AdaGAN: Boosting Generative Models'
Coded by Ilya Tolstikhin, Carl-Johann Simon-Gabriel
"""

import os
import argparse
import logging
import tensorflow as tf
from datahandler import DataHandler
from adagan import AdaGan
from metrics import Metrics
import utils

flags = tf.app.flags
flags.DEFINE_float("g_learning_rate", 0.016,
                   "Learning rate for Generator optimizers [16e-4]")
flags.DEFINE_float("d_learning_rate", 0.0004,
                   "Learning rate for Discriminator optimizers [4e-4]")
flags.DEFINE_float("learning_rate", 0.0008,
                   "Learning rate for other optimizers [8e-4]")
flags.DEFINE_float("adam_beta1", 0.5, "Beta1 parameter for Adam optimizer [0.5]")
flags.DEFINE_integer("zdim", 10, "Dimensionality of the latent space [100]")
flags.DEFINE_float("init_std", 0.02, "Initial variance for weights [0.02]")
flags.DEFINE_string("workdir", 'results', "Working directory ['results']")
flags.DEFINE_bool("use_std_params", True, "Use standard params for this dataset [True]")
flags.DEFINE_bool("unrolled", True, "Use unrolled GAN training [True]")
flags.DEFINE_bool("is_bagging", False, "Do we want to use bagging instead of adagan? [False]")
flags.DEFINE_integer("unrolling_steps", 5, "Number of unrolling steps (0 = usual gan) [5]")
flags.DEFINE_string("objective", 'JS_modified', "Which phi-divergence to use ['JS_modified']")
FLAGS = flags.FLAGS

def main():
    opts = {}
    opts['random_seed'] = 66
    opts['dataset'] = 'mnist3' # gmm, circle_gmm,  mnist, mnist3 ...
    opts['unrolled'] = FLAGS.unrolled # Use Unrolled GAN? (only for images)
    opts['use_std_params'] = FLAGS.use_std_params
    opts['unrolling_steps'] = FLAGS.unrolling_steps # Used only if unrolled = True
    opts['data_dir'] = 'mnist'
    opts['trained_model_path'] = 'models'
    opts['mnist_trained_model_file'] = 'mnist_trainSteps_19999_yhat' # 'mnist_trainSteps_20000'
    opts['gmm_max_val'] = 15.
    opts['toy_dataset_size'] = 10000
    opts['toy_dataset_dim'] = 2
    opts['mnist3_dataset_size'] = 128 # 64 * 2500
    opts['mnist3_to_channels'] = False # Hide 3 digits of MNIST to channels
    opts['input_normalize_sym'] = True # Normalize data to [-1, 1]
    opts['adagan_steps_total'] = 5
    opts['samples_per_component'] = 5000 # 50000
    opts['work_dir'] = FLAGS.workdir
    opts['is_bagging'] = FLAGS.is_bagging
    opts['beta_heur'] = 'uniform' # uniform, constant
    opts['weights_heur'] = 'theory_star' # theory_star, theory_dagger, topk
    opts['beta_constant'] = 0.5
    opts['topk_constant'] = 0.5
    opts["init_std"] = FLAGS.init_std
    opts["init_bias"] = 0.0
    opts['latent_space_distr'] = 'normal' # uniform, normal
    opts['optimizer'] = 'adam' # sgd, adam
    opts["batch_size"] = 64
    opts["d_steps"] = 1
    opts["g_steps"] = 1
    opts["verbose"] = True
    opts['tf_run_batch_size'] = 100

    opts['gmm_modes_num'] = 5
    opts['latent_space_dim'] = FLAGS.zdim
    opts["gan_epoch_num"] = 2
    opts["mixture_c_epoch_num"] = 1
    opts['opt_learning_rate'] = FLAGS.learning_rate
    opts['opt_d_learning_rate'] = FLAGS.d_learning_rate
    opts['opt_g_learning_rate'] = FLAGS.g_learning_rate
    opts["opt_beta1"] = FLAGS.adam_beta1
    opts['batch_norm_eps'] = 1e-05
    opts['batch_norm_decay'] = 0.9
    opts['d_num_filters'] = 16
    opts['g_num_filters'] = 16
    opts['conv_filters_dim'] = 4
    opts["early_stop"] = -1 # set -1 to run normally
    opts["plot_every"] = 1 # 50 # set -1 to run normally
    opts["eval_points_num"] = 3000 # 25600
    opts['digit_classification_threshold'] = 0.999
    opts['objective'] = FLAGS.objective
    opts['inverse_metric'] = False # Use metric from the Unrolled GAN paper?

    if opts['use_std_params']:
        if opts['dataset'] is 'circle_gmm':
            # Standard toyUnrolledGan parameters
            opts['samples_per_component'] = 3000 # 100 # 50000
            opts["plot_every"] = 5 # set -1 to run normally
            opts['d_num_filters'] = 128
            opts['g_num_filters'] = 128
            opts["init_std"] = .2
            opts["batch_size"] = 512
            opts['adagan_steps_total'] = 2
            opts['toy_dataset_size'] = 512 * 5
            opts['toy_dataset_dim'] = 2
            opts['gmm_modes_num'] = 8
            opts['latent_space_dim'] = 256
            opts["gan_epoch_num"] = 5
            opts['opt_d_learning_rate'] = 1e-4
            opts['opt_g_learning_rate'] = 1e-3
            opts["opt_beta1"] = .5
        if opts['dataset'] is 'mnist3':
            opts['latent_space_dim'] = 50
            opts['mnist3_dataset_size'] = 128 # 64 * 2500
            opts['mnist3_to_channels'] = False # Hide 3 digits of MNIST to channels
            opts['adagan_steps_total'] = 5
            opts['samples_per_component'] = 5000 # 50000
            opts["eval_points_num"] = 3000 # 25600
        else:
            pass

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    utils.create_dir(opts['work_dir'])
    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'
    adagan = AdaGan(opts, data)
    metrics = Metrics()

    for step in range(opts["adagan_steps_total"]):
        logging.info('Running step {} of AdaGAN'.format(step + 1))
        adagan.make_step(opts, data)
        num_fake = opts['eval_points_num']
        logging.debug('Sampling fake points')
        fake_points = adagan.sample_mixture(num_fake)
        logging.debug('Sampling more fake points')
        more_fake_points = adagan.sample_mixture(500)
        logging.debug('Plotting results')
        if opts['dataset'] == 'gmm':
            metrics.make_plots(opts, step, data.data[:500],
                    fake_points[0:100], adagan._data_weights[:500])
            logging.debug('Evaluating results')
            (likelihood, C) = metrics.evaluate(
                opts, step, data.data[:500],
                fake_points, more_fake_points, prefix='')
        else:
            metrics.make_plots(opts, step, data.data,
                    fake_points[:4 * 16], adagan._data_weights)
            logging.debug('Evaluating results')
            res = metrics.evaluate(
                opts, step, data.data[:500],
                fake_points, more_fake_points, prefix='')
    logging.debug("AdaGan finished working!")

if __name__ == '__main__':
    main()
