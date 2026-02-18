##################################################
## Project:   sparseVCBART sim
## Date:      Feb 18, 2026
## Author:    Arnie Seong
##################################################


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
  round(n_obs * 1.5), # training obs
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
trn_inds <- 1:n_obs
tst_inds <- (n_obs + 1):round(n_obs * 1.5)
Ey <- Ey_df[,1]
Ey_trn <- Ey[trn_inds]
Ey_tst <- Ey[tst_inds]
XZ_trn <- Ey_df[trn_inds, -1]
XZ_tst <- Ey_df[tst_inds, -1]

eps_mat <- matrix(
  rnorm(
    n = n_sims * round(n_obs * 1.5),
    mean = mu_eps,
    sd = sig_eps
  ), 
  ncol = n_sims
)




# NETWORK SETUP ----

## param counts ----
source(here::here("Rcode", "analysis_fcns.R"))
param_counts_from_dims(dim_vec = c(23, 4, 16, 1))  # first experiment p=3
param_counts_from_dims(dim_vec = c(70, 4, 16, 1))  # second experiment p=50


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
  "spVCBART_"
)














