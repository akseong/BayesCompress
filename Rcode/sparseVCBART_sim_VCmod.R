##################################################
## Project:   sparseVCBART sim - VC structure modeled
## Date:      Feb 18, 2026
## Author:    Arnie Seong
##################################################

# VC structure modeled by BNN (i.e. BNN input is Z, output is coefs)
# - unlike sparseVCBART, not modeling each coef separately


# SETUP: LIBS & FCNS ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe_klcorrected.R"))
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))
source(here("Rcode", "sparseVCBART_fcns.R"))



# GEN DATA ----
# matching to description of synthetic data in sparseVCBART. 
# Experiment 1 uses p=3.  Experiment 2 uses p=50
# HOWEVER: function for \beta_1 described in paper appears to be wrong
# (doesn't even match their own plot).  The plot looks
# much closer to the function for \beta_2 in the orig. VCBART paper
# which is what I'm using instead.

## simdat params ----
n_sims <- 2
tt_ratio <- 1.25


## gendat params ----
n_obs <- 1e3
p <- 3  
R <- 20
sig_eps <- 1
mu_eps <- 0
true_covs <- c(
  paste0("x", 1:3),
  paste0("z", 1:5)
)

## beta_j(Z) functions ----
# beta_j(z) functions in sparseVCBART_fcns.R
bfcns_list <- list(
  "beta_0" = beta_0,
  "beta_1" = beta_1,
  "beta_2" = beta_2,
  "beta_3" = beta_3
)

# plot beta_0, beta_1 fcns
#   note that b1 and b0 are not really separable when looking at y
#   without modeling the effect modifiers directly as VCBART does
#   - to capture "b1" we need to set x1 = 1, generate predictions yhat
#     and then generate intercepts b0hat using the same Z coordinates 
#     and setting all x = 0.  Then, plot (yhat - b0hat) against z1
plot_b0_true(resol = 100, b0 = bfcns_list$beta_0)
plot_b1_true(resol = 100, b1 = bfcns_list$beta_1)

## generate Ey, X ----
# Covariance of X vars (same as in paper)
#   function also in sparseVCBART_fcns.R
#   corr_fcn <- function(i, j) {0.5^(abs(i-j))} 

Ey_df <- gen_Eydat_sparseVCBART(
  round(n_obs * tt_ratio), # training obs
  p,
  R,
  covar_fcn = corr_fcn,
  beta_0 = bfcns_list$beta_0,
  beta_1 = bfcns_list$beta_1,
  beta_2 = bfcns_list$beta_2,
  beta_3 = bfcns_list$beta_3
)

## train/test ----
# Note: (test obs aren't used to calibrate NN,
# just for me to observe for training progress
# and catch simulation problems)
tr_inds <- 1:n_obs
te_inds <- (n_obs + 1):round(n_obs * tt_ratio)
Ey_raw <- Ey_df[,1]
Ey_raw_tr <- Ey_raw[tr_inds]
Ey_raw_te <- Ey_raw[te_inds]
XZ_raw <- Ey_df[, -1]

# standardize XZ.  
# standardizing Y will have to happen after epsilons added
XZ_means <- colMeans(XZ_raw)
XZ_sds <- apply(XZ_raw, 2, sd)
XZ_centered <- sweep(XZ_raw, 2, STATS = XZ_means, "-")
XZ <- sweep(XZ_centered, 2, STATS = XZ_sds, "/")

# # check standardized
# colMeans(XZ)
# diag(cov(XZ))
XZ_tr <- XZ[tr_inds, ]
XZ_te <- XZ[te_inds, ]

# generate epsilons
eps_mat <- matrix(
  rnorm(
    n = n_sims * round(n_obs * tt_ratio),
    mean = mu_eps,
    sd = sig_eps
  ), 
  ncol = n_sims
)


# NETWORK SETUP ----

## param counts ----
param_counts_from_dims(dim_vec = c(23, 4, 8, 8, 1))  # first experiment p=3
param_counts_from_dims(dim_vec = c(70, 4, 8, 8, 1))  # second experiment p=50


## CUDA----
if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}

## save paths ----
save_prestem <- here::here(
  "sims", 
  "results", 
  "spVCBART_exp1_416_"
)

## sim_params
sim_params <- list(
  "sim_name" = "1k obs; sparseVCBART experiment 1: R=20, p=3",
  "seed" = 416,
  "n_sims" = 5, 
  "train_epochs" = 5E5,
  "report_every" = 1E4,
  "use_cuda" = use_cuda,
  "d_in" = R+p,
  "d_hidden1" = 4,
  "d_hidden2" = 16,
  # "d_hidden3" = 16,
  # "d_hidden4" = 16,
  # "d_hidden5" = 16,
  "d_out" = 1,
  "n_obs" = 1250,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "alpha_thresh" = 1 / qchisq(1 - (0.05 / 104), df = 1),
  "flist" = flist,
  "lr" = 0.001,  # sim_hshoe learning rate arg.  If not specified, uses optim_adam default (0.001)
  "err_sig" = 1,
  "xdist" = "norm",
  "convergence_crit" = 1e-7,
  "ttsplit" = 4/5,
  "batch_size" = NULL,
  "stop_k" = 100,
  "stop_streak" = 25,
  "burn_in" = 25e4 # 5E5,
)
set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))














