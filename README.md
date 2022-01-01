BayesCompress for Variable Selection (in progress)
================
Arnie Seong
01 January 2022







# Description

This project is an **(in-progress)** adaptation of [the methods
described here (Louizos, Ullrich, Welling
2017)](https://arxiv.org/abs/1705.08665) for variable selection. The
main advantage of this method over established variable selection
methods is the combination of the expressivity of deep neural networks
with the known behavior of familiar Bayesian sparsity priors.

# Motivation

The Bayesian Compression method outlined by Louizos, Ullrich, and
Welling places scale-mixture sparsity priors (e.g. spike-and-slab and
horseshoe) on *entire rows* of neural network weight layers rather than
on individual weights. In the typical matrix representation of weight
layers, since row *i* of weight layer *j* is used to compute the inpute
corresponding to column *i*’th in weight layer *j* + 1, if row *i* of
layer *j* is omitted, then column *i* of layer *j* + 1 can also be
omitted, resulting in large reductions in the dimensions of the weight
layers. In addition, the posterior weights’ variances can be used to
impute the optimal number of bits required to encode the weights,
leading to further compressibility.

<img src="D:/Arnie/Github/Classes/BayesCompress/walkthrough/mnist_saved/weight0_e.gif" title="layer sparsity over training epochs" alt="layer sparsity over training epochs" width="40%" height="100%" /><img src="D:/Arnie/Github/Classes/BayesCompress/walkthrough/mnist_saved/weight1_e.gif" title="layer sparsity over training epochs" alt="layer sparsity over training epochs" width="40%" height="100%" />

## Variable Selection

The motivation for this project is the insight that, for the first
weight layer, removing columns would correspond to removing features or
covariates — something we would not obtain by merely omitting individual
weights. As formulated in Louizos, Ullrich, and Welling 2017, the method
described above does not lead to principled variable selection in the
usual statistical sense (though it can be used to examine feature
importance on an observation-by-observation basis), because it does not
remove columns in the first weight layer. We propose adding another
level in the hierarchical sparsity prior to accomplish this. From a
statistical point of view, from the Bayesian sparsity prior
specification we obtain variable selection with known operating
characteristics which, from the neural network architecture, should be
robust to misspecification of functional forms and nonlinearities.
