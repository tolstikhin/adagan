# AdaGAN: Boosting Generative Models

This project implements the AdaGAN algorithm, presented in [this paper](https://arxiv.org/abs/1701.02386).

## Getting started

It is assumed that the user runs TensorFlow of version > 12.

Make sure the directory where you run code also contains sub-directories called *mnist* and *models*
containing MNIST datasets and the pre-trained MNIST classifier respectively (provided in this repo).

## Running

Gaussian Mixture Experiments
```
python adagan_gmm.py [options]
```

MNIST Experiments
```
python adagan_mnist.py [options]
```

3-digit MNIST Experiments
```
python adagan_mnist3.py [options]
```

## Code is still under development and can be modified
