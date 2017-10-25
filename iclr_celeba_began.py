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
import numpy as np
from datahandler import DataHandler
from adagan import AdaGan
from metrics import Metrics
import utils

flags = tf.app.flags
flags.DEFINE_float("g_learning_rate", 0.0001,
                   "Learning rate for Generator optimizers [16e-4]")
flags.DEFINE_float("d_learning_rate", 0.00005,
                   "Learning rate for Discriminator optimizers [4e-4]")
flags.DEFINE_float("learning_rate", 0.003,
                   "Learning rate for other optimizers [8e-4]")
flags.DEFINE_float("adam_beta1", 0.5, "Beta1 parameter for Adam optimizer [0.5]")
flags.DEFINE_integer("zdim", 64, "Dimensionality of the latent space [100]")
flags.DEFINE_float("init_std", 0.0099999, "Initial variance for weights [0.02]")
flags.DEFINE_string("workdir", 'results_celeba_pot', "Working directory ['results']")
flags.DEFINE_bool("unrolled", False, "Use unrolled GAN training [True]")
flags.DEFINE_bool("vae", False, "Use VAE instead of GAN")
flags.DEFINE_bool("pot", True, "Use POT instead of GAN")
flags.DEFINE_float("pot_lambda", 10., "POT regularization")
flags.DEFINE_bool("is_bagging", False, "Do we want to use bagging instead of adagan? [False]")
FLAGS = flags.FLAGS

def main():
    opts = {}
    # Utility
    opts['random_seed'] = 66
    opts['dataset'] = 'celebA' # gmm, circle_gmm,  mnist, mnist3 ...
    opts['celebA_crop'] = 'closecrop' # closecrop or resizecrop
    opts['data_dir'] = 'celebA/datasets/celeba/img_align_celeba'
    opts['trained_model_path'] = None #'models'
    opts['mnist_trained_model_file'] = None #'mnist_trainSteps_19999_yhat' # 'mnist_trainSteps_20000'
    opts['work_dir'] = FLAGS.workdir
    opts['ckpt_dir'] = 'checkpoints'
    opts["verbose"] = 2
    opts['tf_run_batch_size'] = 128
    opts["early_stop"] = -1 # set -1 to run normally
    opts["plot_every"] = 500
    opts["save_every_epoch"] = 20
    opts['gmm_max_val'] = 15.

    # Datasets
    opts['toy_dataset_size'] = 10000
    opts['toy_dataset_dim'] = 2
    opts['mnist3_dataset_size'] = 2 * 64 # 64 * 2500
    opts['mnist3_to_channels'] = False # Hide 3 digits of MNIST to channels
    opts['input_normalize_sym'] = True # Normalize data to [-1, 1]
    opts['gmm_modes_num'] = 5

    # AdaGAN parameters
    opts['adagan_steps_total'] = 1
    opts['samples_per_component'] = 1000
    opts['is_bagging'] = FLAGS.is_bagging
    opts['beta_heur'] = 'uniform' # uniform, constant
    opts['weights_heur'] = 'theory_star' # theory_star, theory_dagger, topk
    opts['beta_constant'] = 0.5
    opts['topk_constant'] = 0.5
    opts["mixture_c_epoch_num"] = 5
    opts["eval_points_num"] = 25600
    opts['digit_classification_threshold'] = 0.999
    opts['inverse_metric'] = False # Use metric from the Unrolled GAN paper?
    opts['inverse_num'] = 100 # Number of real points to inverse.
    opts['objective'] = None

    # Generative model parameters
    opts["init_std"] = FLAGS.init_std
    opts["init_bias"] = 0.0
    opts['latent_space_distr'] = 'normal' # uniform, normal
    opts['latent_space_dim'] = FLAGS.zdim
    opts["gan_epoch_num"] = 300
    opts['convolutions'] = True # If False then encoder is MLP of 3 layers
    opts['d_num_filters'] = 1024
    opts['d_num_layers'] = 4
    opts['g_num_filters'] = 256
    opts['g_num_layers'] = 12
    opts['e_is_random'] = False
    opts['e_pretrain'] = True
    opts['e_add_noise'] = True
    opts['e_pretrain_bsize'] = 256
    opts['e_num_filters'] = 256
    opts['e_num_layers'] = 12
    opts['g_arch'] = 'began'
    opts['g_stride1_deconv'] = False
    opts['g_3x3_conv'] = 0
    opts['e_arch'] = 'began'
    opts['e_3x3_conv'] = 0
    opts['conv_filters_dim'] = 3
    # --GAN specific:
    opts['conditional'] = False
    opts['unrolled'] = FLAGS.unrolled # Use Unrolled GAN? (only for images)
    opts['unrolling_steps'] = 5 # Used only if unrolled = True
    # --VAE specific
    opts['vae'] = FLAGS.vae
    opts['vae_sigma'] = 0.01
    # --POT specific
    opts['pot'] = FLAGS.pot
    opts['pot_pz_std'] = 2.
    opts['pot_lambda'] = FLAGS.pot_lambda
    opts['adv_c_loss'] = 'none'
    opts['vgg_layer'] = 'pool2'
    opts['adv_c_patches_size'] = 5
    opts['adv_c_num_units'] = 32
    opts['adv_c_loss_w'] = 1.0
    opts['cross_p_w'] = 0.0
    opts['diag_p_w'] = 0.0
    opts['emb_c_loss_w'] = 1.0
    opts['reconstr_w'] = 1.0
    opts['z_test'] = 'mmd'
    opts['gan_p_trick'] = False
    opts['pz_transform'] = False
    opts['z_test_corr_w'] = 0.0
    opts['z_test_proj_dim'] = 10

    # Optimizer parameters
    opts['optimizer'] = 'adam' # sgd, adam
    opts["batch_size"] = 64
    opts["d_steps"] = 1
    opts['d_new_minibatch'] = False
    opts["g_steps"] = 2
    opts['batch_norm'] = True
    opts['dropout'] = False
    opts['dropout_keep_prob'] = 0.5
    opts['recon_loss'] = 'l2sq'
    # "manual" or number (float or int) giving the number of epochs to divide
    # the learning rate by 10 (converted into an exp decay per epoch).
    opts['decay_schedule'] = 'plateau'
    opts['opt_learning_rate'] = FLAGS.learning_rate
    opts['opt_d_learning_rate'] = FLAGS.d_learning_rate
    opts['opt_g_learning_rate'] = FLAGS.g_learning_rate
    opts["opt_beta1"] = FLAGS.adam_beta1
    opts['batch_norm_eps'] = 1e-05
    opts['batch_norm_decay'] = 0.9

    if opts['e_is_random']:
        assert opts['latent_space_distr'] == 'normal',\
            'Random encoders currently work only with Gaussian Pz'
    # Data augmentation
    opts['data_augm'] = False

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'], opts['ckpt_dir']))

    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'
    adagan = AdaGan(opts, data)
    metrics = Metrics()

    train_size = data.num_points
    random_idx = np.random.choice(train_size, 4*40, replace=False)
    metrics.make_plots(opts, 0, data.data,
            data.data[random_idx], adagan._data_weights, prefix='dataset_')

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
                    fake_points[:320], adagan._data_weights)
            if opts['inverse_metric']:
                logging.debug('Evaluating results')
                l2 = np.min(adagan._invert_losses[:step + 1], axis=0)
                logging.debug('MSE=%.5f, STD=%.5f' % (np.mean(l2), np.std(l2)))
            res = metrics.evaluate(
                opts, step, data.data[:500],
                fake_points, more_fake_points, prefix='')
    logging.debug("AdaGan finished working!")

if __name__ == '__main__':
    main()
