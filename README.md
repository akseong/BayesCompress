BayesCompress for Variable Selection (in progress)
================
Arnie Seong
18 October 2021







# Description

This project is an **(in-progress)** adaptation of [the methods
described here (Louizos, Ullrich, Welling
2017)](https://arxiv.org/abs/1705.08665) for variable selection. The
main advantage of this method over established variable selection
methods is the combination of the expressivity of deep neural networks
with the known behavior of familiar Bayesian sparsity priors.

# Motivation

The Bayesian Compression method outlined by Louizos, Ullrich, and
Welling places scale-mixture sparsity (e.g. spike-and-slab and
horseshoe) priors on *entire rows* of neural network weight layers
rather than on individual weights, allowing for compact representation
of the weight layers since entire rows can be omitted. In addition,
since the input for row *m* of layer *l* is calculated using column *m*
of layer *l-1*, corresponding columns from the previous layer can be
omitted.

<img src="D:/Arnie/Github/Classes/BayesCompress/walkthrough/mnist_saved/weight0_e.gif" title="layer sparsity over training epochs" alt="layer sparsity over training epochs" width="40%" height="100%" /><img src="D:/Arnie/Github/Classes/BayesCompress/walkthrough/mnist_saved/weight1_e.gif" title="layer sparsity over training epochs" alt="layer sparsity over training epochs" width="40%" height="100%" />

The motivation for this project is the insight that, for the input
layer, this pruning scheme corresponds to removing features or
covariates — something we would not obtain by merely omitting individual
weights. From a statistical point of view, from the Bayesian sparsity
prior specification we obtain variable selection with known operating
characteristics which, from the neural network architecture, should be
robust to misspecification of functional forms and nonlinearity.
