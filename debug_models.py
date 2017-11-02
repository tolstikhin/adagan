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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import utils

z_dim = 64
pz_std = 2.
dataset = 'celebA'
num_cols = 10
num_pairs = 10
ckpt_dir = os.path.join('.', 'trained_celeba_gan')
output_dir = 'trained_celeba_gan/pics'
normalyze = True

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(
        # os.path.join('.', 'results_cifar10_pot_conv', 'checkpoints', 'trained-pot-1.meta'))
        os.path.join(ckpt_dir, 'trained-pot-126480.meta'))
    saver.restore(sess, os.path.join(ckpt_dir, 'trained-pot-126480'))
    # saver.restore(sess, os.path.join('.', 'results_cifar10_pot_conv', 'checkpoints', 'trained-pot-1'))
    noise_ph = tf.get_collection('noise_ph')[0]
    bn_ph = tf.get_collection('is_training_ph')[0]
    decoder = tf.get_collection('decoder')[0]

    mean = np.zeros(z_dim)
    cov = np.identity(z_dim)
    noise = pz_std * np.random.multivariate_normal(
        mean, cov, 16 * num_cols).astype(np.float32)

    # 1. Random samples
    res = sess.run(decoder, feed_dict={noise_ph: noise, bn_ph: False})
    metrics = Metrics()
    opts = {}
    opts['dataset'] = dataset
    opts['input_normalize_sym'] = normalyze
    opts['work_dir'] = output_dir
    metrics.make_plots(opts, 0, None, res, prefix='samples')

    # #2. Interpolations
    # ids = np.random.choice(16 * num_cols, num_pairs, replace=False)
    # for i in range(len(ids)):
    #     for j in range(i + 1, len(ids)):
    #         id1, id2 = ids[i], ids[j]
    #         a = np.reshape(noise[id1, :], (1, z_dim))
    #         b = np.reshape(noise[id2, :], (1, z_dim))
    #         _lambda = np.linspace(0., 1., 60)
    #         _lambda = np.reshape(_lambda, (60, 1))
    #         line = np.dot(_lambda, a) + np.dot((1 - _lambda), b)
    #         res = sess.run(decoder, feed_dict={noise_ph: line, bn_ph: False})
    #         metrics = Metrics()
    #         opts = {}
    #         opts['dataset'] = 'mnist'
    #         opts['input_normalize_sym'] = False
    #         opts['work_dir'] = output_dir
    #         metrics.make_plots(opts, 0, None, res, prefix='line%d%d' % (id1, id2), max_rows=1)

    #2. Interpolations
    ids = np.random.choice(16 * num_cols, num_pairs, replace=False)
    res = None
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            a = np.reshape(noise[id1, :], (1, z_dim))
            b = np.reshape(noise[id2, :], (1, z_dim))
            _lambda = np.linspace(0., 1., 60)
            _lambda = np.reshape(_lambda, (60, 1))
            line = np.dot(_lambda, a) + np.dot((1 - _lambda), b)
            pics = sess.run(decoder, feed_dict={noise_ph: line, bn_ph: False})
            if normalyze:
                pics = (pics + 1.) / 2.
            if res is None:
                if opts['dataset'] == 'mnist':
                    res = 1. - pics
                else:
                    res = pics
            else:
                if opts['dataset'] == 'mnist':
                    res = np.vstack([res, 1. - pics])
                else:
                    res = np.vstack([res, pics])
    # 60 cols and res.shape[0] rows
    num_cols = 60
    res = np.split(res, num_pairs * (num_pairs - 1) / 2)
    res = np.concatenate(res, axis = 1)
    res = np.concatenate(res, axis = 1)
    if opts['dataset'] == 'mnist':
        image = res[:, :, 0]
    else:
        image = res[:, :, :]

    dpi = 100
    height_pic = image.shape[0]
    width_pic = image.shape[1]
    height = 3 * height_pic / float(dpi)
    width = 3 * width_pic / float(dpi)
    fig = plt.figure(figsize=(width, height))#, dpi=1)
    if opts['dataset'] == 'mnist':
        ax = plt.imshow(image, cmap='Greys', interpolation='none')
    else:
        ax = plt.imshow(image, interpolation='none')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.set_xlim([0, width_pic])
    ax.axes.set_ylim([height_pic, 0])
    ax.axes.set_aspect(1)
    filename = 'interpolations.png'
    fig.savefig(utils.o_gfile((output_dir, filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()


    #3. Random directions
    for i in range(len(ids)):
        id1 = ids[i]
        b = np.reshape(noise[id1, :], (1, z_dim))
        b /= np.sqrt(np.sum(b * b))
        _lambda = np.linspace(0., np.sqrt(z_dim * pz_std * pz_std) * 5, 30)
        _lambda = np.reshape(_lambda, (30, 1))
        line = np.dot(_lambda, b)
        res = sess.run(decoder, feed_dict={noise_ph: line, bn_ph: False})
        metrics = Metrics()
        opts = {}
        opts['dataset'] = dataset
        opts['input_normalize_sym'] = normalyze
        opts['work_dir'] = output_dir
        metrics.make_plots(opts, 0, None, res, prefix='origin%d' % id1, max_rows=1)


