# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""Debugging trained models from checkpoints.

"""

import os
import tensorflow as tf
import numpy as np
import ops
from metrics import Metrics

z_dim = 20
num_cols = 40
num_pairs = 5
ckpt_dir = os.path.join('.', 'trained_cifar')
output_dir = 'test'

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(
        # os.path.join('.', 'results_cifar10_pot_conv', 'checkpoints', 'trained-pot-1.meta'))
        os.path.join(ckpt_dir, 'trained-pot-30000.meta'))
    saver.restore(sess, os.path.join(ckpt_dir, 'trained-pot-30000'))
    # saver.restore(sess, os.path.join('.', 'results_cifar10_pot_conv', 'checkpoints', 'trained-pot-1'))
    noise_ph = tf.get_collection('noise_ph')[0]
    decoder = tf.get_collection('decoder')[0]

    mean = np.zeros(z_dim)
    cov = np.identity(z_dim)
    noise = 2. * np.random.multivariate_normal(
        mean, cov, 16 * num_cols).astype(np.float32)

    # 1. Random samples
    res = sess.run(decoder, feed_dict={noise_ph: noise})
    metrics = Metrics()
    opts = {}
    opts['dataset'] = 'cifar10'
    opts['input_normalize_sym'] = False
    opts['work_dir'] = output_dir
    metrics.make_plots(opts, 0, None, res, prefix='samples')

    #2. Interpolations
    ids = np.random.choice(16 * num_cols, num_pairs, replace=False)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            a = np.reshape(noise[id1, :], (1, z_dim))
            b = np.reshape(noise[id2, :], (1, z_dim))
            _lambda = np.linspace(0., 1., 60)
            _lambda = np.reshape(_lambda, (60, 1))
            line = np.dot(_lambda, a) + np.dot((1 - _lambda), b)
            res = sess.run(decoder, feed_dict={noise_ph: line})
            metrics = Metrics()
            opts = {}
            opts['dataset'] = 'cifar10'
            opts['input_normalize_sym'] = False
            opts['work_dir'] = output_dir
            metrics.make_plots(opts, 0, None, res, prefix='line%d%d' % (id1, id2), max_rows=1)

    #3. Random directions
    for i in range(len(ids)):
        id1 = ids[i]
        b = np.reshape(noise[id1, :], (1, z_dim))
        _lambda = np.linspace(0., 10., 60)
        _lambda = np.reshape(_lambda, (60, 1))
        line = np.dot(_lambda, b)
        res = sess.run(decoder, feed_dict={noise_ph: line})
        metrics = Metrics()
        opts = {}
        opts['dataset'] = 'cifar10'
        opts['input_normalize_sym'] = False
        opts['work_dir'] = output_dir
        metrics.make_plots(opts, 0, None, res, prefix='origin%d' % id1, max_rows=1)


