"""Various utilities.

"""

import os
import sys
import copy
import numpy as np
import logging

def generate_noise(opts, num=100):
    """Generate latent noise.

    """
    noise = None
    if opts['latent_space_distr'] == 'uniform':
        noise = np.random.uniform(
            -1, 1, [num, opts["latent_space_dim"]]).astype(np.float32)
    elif opts['latent_space_distr'] == 'normal':
        mean = np.zeros(opts["latent_space_dim"])
        cov = np.identity(opts["latent_space_dim"])
        noise = np.random.multivariate_normal(
            mean, cov, num).astype(np.float32)
    return noise

class ArraySaver(object):
    """A simple class helping with saving/loading numpy arrays from files.

    This class allows to save / load numpy arrays, while storing them either
    on disk or in memory.
    """

    def __init__(self, mode='ram', workdir=None):
        self._mode = mode
        self._workdir = workdir
        self._global_arrays = {}

    def save(self, name, array):
        if self._mode == 'ram':
            self._global_arrays[name] = copy.deepcopy(array)
        elif self._mode == 'disk':
            create_dir(self._workdir)
            np.save(os.path.join(self._workdir, name), array)
        else:
            assert False, 'Unknown save / load mode'

    def load(self, name):
        if self._mode == 'ram':
            return self._global_arrays[name]
        elif self._mode == 'disk':
            return np.load(os.path.join(self._workdir, name))
        else:
            assert False, 'Unknown save / load mode'

class ProgressBar(object):
    """Super-simple progress bar.

    Thanks to http://stackoverflow.com/questions/3160699/python-progress-bar
    """
    def __init__(self, verbose, iter_num):
        self._width = iter_num
        self.verbose = verbose
        if self.verbose:
            sys.stdout.write("[%s]" % (" " * self._width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self._width + 1))

    def bam(self):
        if self.verbose:
            sys.stdout.write("*")
            sys.stdout.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.verbose:
            sys.stdout.write("\n")

def create_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
