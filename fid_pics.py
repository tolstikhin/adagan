# This script creates plots for experiments from
# ICLR 2018 submission by Bousquet, Gelly, Tolstikhin, Bernhard.
# 1. Random samples
# 2. Interpolations:
#   a. Between points of the test set, linearly
#   b. Take a random point from Pz and make a whole circle on geodesic
# 3. Test reconstructions

import os
import sys
import tensorflow as tf
import numpy as np
import ops
from metrics import Metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import utils
from datahandler import DataHandler

NUM_PICS = 10000
SAVE_REAL_PICS = True
SAVE_PNG = True
SAVE_FAKE_PICS = False
CELEBA_DATA_DIR = 'celebA/datasets/celeba/img_align_celeba'
MNIST_DATA_DIR = 'mnist'
OUT_DIR = 'fid_pics_celeba'

class ExpInfo(object):
    def __init__(self):
        self.trained_model_path = None
        self.model_id = None
        self.pz_std = None
        self.z_dim = None
        self.symmetrize = None
        self.dataset = None
        self.alias = None
        self.test_size = None

def main():
    exp_name = sys.argv[-1]
    create_dir(OUT_DIR)

    exp_names = ['mnist_gan', 'mnist_mmd', 'celeba_gan', 'celeba_mmd']
    cluster_mnist_mmd_path = './mount/GANs/results_mnist_pot_sota_worst2d_plateau_mmd_tricks_expC_81'
    cluster_mnist_mmd2d_path = './mount/GANs/results_mnist_pot_smaller_zdim2_21'
    cluster_mnist_gan_path = './mount/GANs/results_mnist_pot_sota_worst2d1'
    cluster_mnist_vae_path = './mount/GANs/results_mnist_vae_81'
    cluster_mnist_vae2d_path = './mount/GANs/results_mnist_vae_zdim2_21'
    cluster_celeba_gan_path = './mount/GANs/results_celeba_pot_worst2d_plateau_gan_jsmod_641'
    cluster_celeba_vae_path = './mount/GANs/results_celeba_vae_641'
    cluster_celeba_mmd_path = './mount/GANs/results_celeba_pot_worst2d_plateau_mmd_642'
    cluster_celeba_mmd_began_path = './mount/GANs/results_celeba_pot_worst2d_plateau_mmd_began_642'
    model_name_prefix = 'trained-pot-'

    # Exp 1: CelebA with WAE+MMD on 64 dimensional Z space, DCGAN architecture
    exp1 = ExpInfo()
    exp1.trained_model_path = cluster_celeba_mmd_path
    exp1.model_id = '378720'
    exp1.pz_std = 2.0
    exp1.z_dim = 64
    exp1.symmetrize = True
    exp1.dataset = 'celebA'
    exp1.alias = 'celeba_mmd_dcgan'
    exp1.test_size = 512

    # Exp 2: CelebA with WAE+GAN on 64 dimensional Z space, DCGAN architecture
    exp2 = ExpInfo()
    exp2.trained_model_path = cluster_celeba_gan_path
    exp2.model_id = '126480'
    exp2.pz_std = 2.0
    exp2.z_dim = 64
    exp2.symmetrize = True
    exp2.dataset = 'celebA'
    exp2.alias = 'celeba_gan_dcgan'
    exp2.test_size = 512

    # Exp 3: MNIST with WAE+MMD on 8 dimensional Z space, DCGAN architecture
    exp3 = ExpInfo()
    exp3.trained_model_path = cluster_mnist_mmd_path
    exp3.model_id = '55200'
    exp3.pz_std = 1.0
    exp3.z_dim = 8
    exp3.symmetrize = False
    exp3.dataset = 'mnist'
    exp3.alias = 'mnist_mmd_dcgan'
    exp3.test_size = 1000

    # Exp 4: MNIST with WAE+GAN on 8 dimensional Z space, DCGAN architecture
    exp4 = ExpInfo()
    exp4.trained_model_path = cluster_mnist_gan_path
    exp4.model_id = '62100'
    exp4.pz_std = 2.0
    exp4.z_dim = 8
    exp4.symmetrize = False
    exp4.dataset = 'mnist'
    exp4.alias = 'mnist_gan_dcgan'
    exp4.test_size = 1000

    # Exp 5: CelebA with WAE+MMD on 64 dimensional Z space, BEGAN architecture
    exp5 = ExpInfo()
    exp5.trained_model_path = cluster_celeba_mmd_began_path
    exp5.model_id = '157800'
    exp5.pz_std = 2.0
    exp5.z_dim = 64
    exp5.symmetrize = True
    exp5.dataset = 'celebA'
    exp5.alias = 'celeba_mmd_began'
    exp5.test_size = 512

    # Exp 6: MNIST with VAE on 8 dimensional Z space, DCGAN architecture
    exp6 = ExpInfo()
    exp6.trained_model_path = cluster_mnist_vae_path
    exp6.model_id = 'final-69000'
    exp6.pz_std = 1.0
    exp6.z_dim = 8
    exp6.symmetrize = False
    exp6.dataset = 'mnist'
    exp6.alias = 'mnist_vae_dcgan'
    exp6.test_size = 1000

    # Exp 7: MNIST with VAE on 2 dimensional Z space, DCGAN architecture
    exp7 = ExpInfo()
    exp7.trained_model_path = cluster_mnist_vae2d_path
    exp7.model_id = 'final-69000'
    exp7.pz_std = 1.0
    exp7.z_dim = 2
    exp7.symmetrize = False
    exp7.dataset = 'mnist'
    exp7.alias = 'mnist_vae_2d_dcgan'
    exp7.test_size = 1000

    # Exp 8: MNIST with WAE-MMD on 2 dimensional Z space, DCGAN architecture
    exp8 = ExpInfo()
    exp8.trained_model_path = cluster_mnist_mmd2d_path
    exp8.model_id = 'final-69000'
    exp8.pz_std = 2.0
    exp8.z_dim = 2
    exp8.symmetrize = False
    exp8.dataset = 'mnist'
    exp8.alias = 'mnist_mmd_2d_dcgan'
    exp8.test_size = 1000

    # Exp 9: CelebA with VAE on 64 dimensional Z space, dcgan architecture
    exp9 = ExpInfo()
    exp9.trained_model_path = cluster_celeba_vae_path
    exp9.model_id = '126240'
    exp9.pz_std = 1.0
    exp9.z_dim = 64
    exp9.symmetrize = True
    exp9.dataset = 'celebA'
    exp9.alias = 'celeba_vae'
    exp9.test_size = 512


    if exp_name == 'celeba_mmd_dcgan':
        exp = exp1
    elif exp_name == 'celeba_gan_dcgan':
        exp = exp2
    elif exp_name == 'mnist_mmd_dcgan':
        exp = exp3
    elif exp_name == 'mnist_gan_dcgan':
        exp = exp4
    elif exp_name == 'celeba_mmd_began':
        exp = exp5
    elif exp_name == 'mnist_vae':
        exp = exp6
    elif exp_name == 'mnist_vae_2d':
        exp = exp7
    elif exp_name == 'mnist_mmd_2d':
        exp = exp8
    elif exp_name == 'celeba_vae':
        exp = exp9

    exp_list = [exp]

    for exp in exp_list:

        output_dir = os.path.join(OUT_DIR, exp.alias)
        create_dir(output_dir)

        z_dim = exp.z_dim
        pz_std = exp.pz_std
        dataset = exp.dataset
        model_path = exp.trained_model_path
        normalyze = exp.symmetrize

        if SAVE_REAL_PICS:
            pic_dir = os.path.join(output_dir, 'real')
            create_dir(pic_dir)
            # Saving real pics
            opts = {}
            opts['dataset'] = dataset
            opts['input_normalize_sym'] = normalyze
            opts['work_dir'] = output_dir
            if exp.dataset == 'celebA':
                opts['data_dir'] = CELEBA_DATA_DIR
            elif exp.dataset == 'mnist':
                opts['data_dir'] = MNIST_DATA_DIR
            opts['celebA_crop'] = 'closecrop'
            data = DataHandler(opts)
            pic_id = 1
            if dataset == 'celebA':
                shuffled_ids = np.load(os.path.join(model_path, 'shuffled_training_ids'))
                test_ids = shuffled_ids[-exp.test_size:]
                test_images = data.data
                train_ids = shuffled_ids[:-exp.test_size]
                train_images = data.data
            else:
                test_images = data.test_data.X
                train_images = data.data.X
                train_ids = range(len(train_images))
                test_ids = range(len(test_images))
            if SAVE_PNG:
                for idx in test_ids:
                    if pic_id % 1000 == 0:
                        print 'Saved %d/%d' % (pic_id, NUM_PICS)
                    save_pic(test_images[idx], os.path.join(pic_dir, 'real_image{:05d}.png'.format(pic_id)), exp)
                    pic_id += 1
                    if pic_id > NUM_PICS:
                        break
            num_remain = max(NUM_PICS - len(test_ids), 0)
            train_size = data.num_points
            rand_train_ids = np.random.choice(train_size, num_remain, replace=False)
            rand_train_ids = [train_ids[idx] for idx in rand_train_ids]
            rand_train_pics = train_images[rand_train_ids]
            if SAVE_PNG:
                for i in range(num_remain):
                    if pic_id % 1000 == 0:
                        print 'Saved %d/%d' % (pic_id, NUM_PICS)
                    save_pic(rand_train_pics[i], os.path.join(pic_dir, 'real_image{:05d}.png'.format(pic_id)), exp)
                    pic_id += 1
            all_pics = np.vstack([test_images, rand_train_pics])
            all_pics = all_pics.astype(np.float)
            if len(all_pics) > NUM_PICS:
                all_pics = all_pics[:NUM_PICS]
            np.random.shuffle(all_pics)
            np.save(os.path.join(output_dir, 'real'), all_pics)


        if SAVE_FAKE_PICS:
            with tf.Session() as sess:
                with sess.graph.as_default():
                    saver = tf.train.import_meta_graph(
                        os.path.join(model_path, 'checkpoints', model_name_prefix + exp.model_id + '.meta'))
                    saver.restore(sess, os.path.join(model_path, 'checkpoints', model_name_prefix + exp.model_id))
                    real_points_ph = tf.get_collection('real_points_ph')[0]
                    noise_ph = tf.get_collection('noise_ph')[0]
                    is_training_ph = tf.get_collection('is_training_ph')[0]
                    decoder = tf.get_collection('decoder')[0]

                    # Saving random samples
                    mean = np.zeros(z_dim)
                    cov = np.identity(z_dim)
                    noise = pz_std * np.random.multivariate_normal(
                        mean, cov, NUM_PICS).astype(np.float32)
                    res = sess.run(decoder, feed_dict={noise_ph: noise, is_training_ph: False})
                    pic_dir = os.path.join(output_dir, 'fake')
                    create_dir(pic_dir)
                    if SAVE_PNG:
                        for i in range(1, NUM_PICS + 1):
                            if i % 1000 == 0:
                                print 'Saved %d/%d' % (i, NUM_PICS)
                            save_pic(res[i-1], os.path.join(pic_dir, 'fake_image{:05d}.png'.format(i)), exp)
                    np.save(os.path.join(output_dir, 'fake'), res)

def save_pic(pic, path, exp):
    if len(pic.shape) == 4:
        pic = pic[0]
    height = pic.shape[0]
    width = pic.shape[1]
    fig = plt.figure(frameon=False, figsize=(width, height))#, dpi=1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if exp.symmetrize:
        pic = (pic + 1.) / 2.
    if exp.dataset == 'mnist':
        pic = pic[:, :, 0]
        pic = 1. - pic
    if exp.dataset == 'mnist':
        ax.imshow(pic, cmap='Greys', interpolation='none')
    else:
        ax.imshow(pic, interpolation='none')
    fig.savefig(path, dpi=1, format='png')
    plt.close()
    # if exp.dataset == 'mnist':
    #     pic = pic[:, :, 0]
    #     pic = 1. - pic
    #     ax = plt.imshow(pic, cmap='Greys', interpolation='none')
    # else:
    #     ax = plt.imshow(pic, interpolation='none')
    # ax.axes.get_xaxis().set_ticks([])
    # ax.axes.get_yaxis().set_ticks([])
    # ax.axes.set_xlim([0, width])
    # ax.axes.set_ylim([height, 0])
    # ax.axes.set_aspect(1)
    # fig.savefig(path, format='png')
    # plt.close()

def create_dir(d):
    if not tf.gfile.IsDirectory(d):
        tf.gfile.MakeDirs(d)

main()
