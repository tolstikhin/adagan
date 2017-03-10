# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os
import logging
import numpy as np
import utils
from PIL import Image

class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """
    def __init__(self, opts):
        self.data_shape = None
        self.num_points = None
        self.data = None
        self.labels = None
        self._load_data(opts)

    def _load_data(self, opts):
        """Load a dataset and fill all the necessary variables.

        """
        if opts['dataset'] == 'mnist':
            self._load_mnist(opts)
        if opts['dataset'] == 'mnist3':
            self._load_mnist3(opts)
        if opts['dataset'] == 'gmm':
            self._load_gmm(opts)
        if opts['dataset'] == 'circle_gmm':
            self._load_mog(opts)
        if opts['dataset'] == 'guitars':
            self._load_guitars(opts)

        if opts['input_normalize_sym'] and  \
                opts['dataset'] in ('mnist', 'mnist3', 'guitars'):
            # Normalize data to [-1, 1]
            self.data = (self.data - 0.5) * 2.

    def _load_mog(self, opts):
        """Sample data from the mixture of Gaussians on circle.

        """

        # Only use this setting in dimension 2
        assert opts['toy_dataset_dim'] == 2

        # First we choose parameters of gmm and thus seed
        radius = opts['gmm_max_val']
        modes_num = opts["gmm_modes_num"]
        np.random.seed(opts["random_seed"])

        thetas = np.linspace(0, 2 * np.pi, modes_num)
        mixture_means = np.stack((radius * np.sin(thetas), radius * np.cos(thetas)), axis=1)
        mixture_variance = 0.01

        # Now we sample points, for that we unseed
        np.random.seed()
        num = opts['toy_dataset_size']
        X = np.zeros((num, opts['toy_dataset_dim'], 1, 1))
        for idx in xrange(num):
            comp_id = np.random.randint(modes_num)
            mean = mixture_means[comp_id]
            cov = mixture_variance * np.identity(opts["toy_dataset_dim"])
            X[idx, :, 0, 0] = np.random.multivariate_normal(mean, cov, 1)

        self.data_shape = (opts['toy_dataset_dim'], 1, 1)
        self.data = X
        self.num_points = len(X)

    def _load_gmm(self, opts):
        """Sample data from the mixture of Gaussians.

        """

        logging.debug('Loading GMM dataset...')
        # First we choose parameters of gmm and thus seed
        modes_num = opts["gmm_modes_num"]
        np.random.seed(opts["random_seed"])
        max_val = opts['gmm_max_val']
        mixture_means = np.random.uniform(
            low=-max_val, high=max_val,
            size=(modes_num, opts['toy_dataset_dim']))

        def variance_factor(num, dim):
            if num == 1: return 3 ** (2. / dim)
            if num == 2: return 3 ** (2. / dim)
            if num == 3: return 8 ** (2. / dim)
            if num == 4: return 20 ** (2. / dim)
            if num == 5: return 10 ** (2. / dim)
            return num ** 2.0 * 3

        mixture_variance = \
                max_val / variance_factor(modes_num, opts['toy_dataset_dim'])

        # Now we sample points, for that we unseed
        np.random.seed()
        num = opts['toy_dataset_size']
        X = np.zeros((num, opts['toy_dataset_dim'], 1, 1))
        for idx in xrange(num):
            comp_id = np.random.randint(modes_num)
            mean = mixture_means[comp_id]
            cov = mixture_variance * np.identity(opts["toy_dataset_dim"])
            X[idx, :, 0, 0] = np.random.multivariate_normal(mean, cov, 1)

        self.data_shape = (opts['toy_dataset_dim'], 1, 1)
        self.data = X
        self.num_points = len(X)

        logging.debug('Loading GMM dataset done!')

    def _load_guitars(self, opts):
        """Load data from Thomann files.

        """
        logging.debug('Loading Guitars dataset')
        data_dir = os.path.join('./', 'thomann')
        X = None
        files = utils.listdir(data_dir)
        pics = []
        for f in sorted(files):
            if '.jpg' in f and f[0] != '.':
                im = Image.open(utils.o_gfile((data_dir, f), 'rb'))
                res = np.array(im.getdata()).reshape(128, 128, 3)
                pics.append(res)
        X = np.array(pics)

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed()

        self.data_shape = (128, 128, 3)
        self.data = X/255.
        self.num_points = len(X)

        logging.debug('Loading Done.')

    def _load_mnist(self, opts):
        """Load data from MNIST files.

        """
        logging.debug('Loading MNIST')
        data_dir = os.path.join('./', opts['data_dir'])
        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        with utils.o_gfile((data_dir, 'train-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 'train-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_Y = loaded[8:].reshape((60000)).astype(np.int)

        with utils.o_gfile((data_dir, 't10k-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_X = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 't10k-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_Y = loaded[8:].reshape((10000)).astype(np.int)

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)

        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_Y, te_Y), axis=0)

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.seed()

        self.data_shape = (28, 28, 1)
        self.data = X / 255.
        self.labels = y
        self.num_points = len(X)

        logging.debug('Loading Done.')

    def _load_mnist3(self, opts):
        """Load data from MNIST files.

        """
        logging.debug('Loading 3-digit MNIST')
        data_dir = os.path.join('./', opts['data_dir'])
        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        with utils.o_gfile((data_dir, 'train-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 'train-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_Y = loaded[8:].reshape((60000)).astype(np.int)

        with utils.o_gfile((data_dir, 't10k-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_X = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 't10k-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_Y = loaded[8:].reshape((10000)).astype(np.int)

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)

        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_Y, te_Y), axis=0)

        num = opts['mnist3_dataset_size']
        ids = np.random.choice(len(X), (num, 3), replace=True)
        if opts['mnist3_to_channels']:
            # Concatenate 3 digits ito 3 channels
            X3 = np.zeros((num, 28, 28, 3))
            y3 = np.zeros(num)
            for idx, _id in enumerate(ids):
                X3[idx, :, :, 0] = np.squeeze(X[_id[0]], axis=2)
                X3[idx, :, :, 1] = np.squeeze(X[_id[1]], axis=2)
                X3[idx, :, :, 2] = np.squeeze(X[_id[2]], axis=2)
                y3[idx] = y[_id[0]] * 100 + y[_id[1]] * 10 + y[_id[2]]
            self.data_shape = (28, 28, 3)
        else:
            # Concatenate 3 digits in width
            X3 = np.zeros((num, 28, 3 * 28, 1))
            y3 = np.zeros(num)
            for idx, _id in enumerate(ids):
                X3[idx, :, 0:28, 0] = np.squeeze(X[_id[0]], axis=2)
                X3[idx, :, 28:56, 0] = np.squeeze(X[_id[1]], axis=2)
                X3[idx, :, 56:84, 0] = np.squeeze(X[_id[2]], axis=2)
                y3[idx] = y[_id[0]] * 100 + y[_id[1]] * 10 + y[_id[2]]
            self.data_shape = (28, 28 * 3, 1)

        self.data = X3/255.
        y3 = y3.astype(int)
        self.labels = y3
        self.num_points = num

        logging.debug('Training set JS=%.4f' % utils.js_div_uniform(y3))
        logging.debug('Loading Done.')
