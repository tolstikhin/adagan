"""Training AdaGAN on various datasets.

Refer to the arXiv paper 'AdaGAN: Boosting Generative Models'
Coded by Ilya Tolstikhin, Carl-Johann Simon-Gabriel
"""

import argparse
import logging
import tensorflow as tf
from datahandler import DataHandler
from adagan import AdaGan
from metrics import Metrics

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0008, "Learning rate for optimizers [8e-4]")
flags.DEFINE_float("adam_beta1", 0.5, "Beta1 parameter for Adam optimizer [0.5]")
flags.DEFINE_integer("zdim", 256, "Dimensionality of the latent space [100]")
flags.DEFINE_float("init_std", 0.02, "Initial variance for weights [0.02]")
flags.DEFINE_string("workdir", 'results', "Working directory ['results']")
FLAGS = flags.FLAGS

def main():
    opts = {}
    opts['random_seed'] = 66
    opts['dataset'] = 'mnist3' # gmm, mnist, mnist3 ...
    opts['data_dir'] = 'mnist'
    opts['trained_model_path'] = 'models'
    opts['mnist_trained_model_file'] = 'mnist_trainSteps_20000'
    opts['gmm_max_val'] = 15.
    opts['toy_dataset_size'] = 10000
    opts['toy_dataset_dim'] = 2
    opts['mnist3_dataset_size'] = 64 * 1000
    opts['adagan_steps_total'] = 1
    opts['samples_per_component'] = 10000
    opts['work_dir'] = FLAGS.workdir
    opts['is_bagging'] = False
    opts['beta_heur'] = 'constant' # uniform, constant
    opts['weights_heur'] = 'theory_star' # theory_star, theory_dagger, topk
    opts['beta_constant'] = 0.5
    opts['topk_constant'] = 0.5
    opts["init_std"] = FLAGS.init_std
    opts["init_bias"] = 0.0
    opts['latent_space_distr'] = 'normal' # uniform, normal
    opts['optimizer'] = 'adam' # sgd, adam
    opts["batch_size"] = 64
    opts["d_steps"] = 1
    opts["g_steps"] = 2
    opts["verbose"] = True
    opts['tf_run_batch_size'] = 100

    opts['gmm_modes_num'] = 5
    opts['latent_space_dim'] = FLAGS.zdim
    opts["gan_epoch_num"] = 20
    opts["mixture_c_epoch_num"] = 1
    opts['opt_learning_rate'] = FLAGS.learning_rate
    opts["opt_beta1"] = FLAGS.adam_beta1
    opts['batch_norm_eps'] = 1e-05
    opts['batch_norm_decay'] = 0.9
    opts['conv_filters_dim'] = 4
    opts["early_stop"] = -1 # set -1 to run normally

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    data = DataHandler(opts)
    adagan = AdaGan(opts, data)
    metrics = Metrics()

    for step in range(opts["adagan_steps_total"]):
        logging.info('Running step {} of AdaGAN'.format(step + 1))
        adagan.make_step(opts, data)
        fake_points = adagan.sample_mixture(500)
        more_fake_points = adagan.sample_mixture(500)
        metrics.make_plots(opts, step, data.data[:500],
                           fake_points, adagan._data_weights[:500])
        (likelihood, C) = metrics.evaluate(
            opts, step, data.data[:500],
            fake_points, more_fake_points, prefix='')


if __name__ == '__main__':
    main()
