##################################################
## Project:   sparseVCBART sim - VC structure modeled
## Date:      Feb 18, 2026
## Author:    Arnie Seong
##################################################

# VC structure modeled by BNN (i.e. BNN input is Z, output is coefs)
# - unlike sparseVCBART, not modeling each coef separately


# SETUP: LIBS & FCNS ----
library(here)
library(tidyverse)

library(torch)
source(here("Rcode", "torch_horseshoe_klcorrected.R"))
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))
source(here("Rcode", "sparseVCBART_fcns.R"))

# CUDA----
if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}


# data characteristics ----
n_obs <- 1e4   # try with more obs for now
ttsplit <- 0.8
p <- 3  
R <- 20
sig_eps <- 1
mu_eps <- 0
true_covs <- c(
  paste0("x", 1:3),
  paste0("z", 1:5)
)


# SIM PARAMS ----
n_sims <- 2
p_0 <- (p+R)/2
dont_scale_t0 <- TRUE
sim_ID <- "VC_vanilla816_agnostic"

fname_stem <- here::here(
  "sims", 
  "results", 
  paste0(
    sim_ID,
    "_p", p,
    "_n", n_obs/1000, "k",
    "_"
  )
)

## sim_params ** ----
sim_params <- list(
  "description" = "agnostic tau_0; sparseVCBART experiment 1 setting",
  "seed" = 816,
  "n_sims" = 5, 
  "train_epochs" = 5E5,
  "report_every" = 1E4,
  "use_cuda" = use_cuda,
  "d_0" = R+p,
  "d_1" = 8,
  "d_2" = 16,
  # "d_3" = 16,
  # "d_4" = 16,
  # "d_5" = 16,
  "d_L" = 1,
  "n_obs" = n_obs,
  "lr" = 0.001,  # sim_hshoe learning rate arg.  If not specified, uses optim_adam default (0.001)
  "err_sig" = 1,
  "ttsplit" = ttsplit,
  "dont_scale_t0" = dont_scale_t0
)
set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))

## param count ----
dim_vec <- do.call(c, sim_params[grep(pattern = "d_", names(sim_params))])
param_count <- param_counts_from_dims(dim_vec)
cat("\n network parameter count: ") 
print(param_count)


## calibrate tau_0 ** ----
# Piironen & Vehtari 2017 suggest tau_0 = p_0 / (d - p_0) * sig / sqrt(n)
# where p_0 = prior estimate of number of nonzero betas, d = total number of covs

# If many more network params than obs (e.g. like 2x), 
# can try scaling the prior tau by n_obs/n_params
# to induce more shrinkage (put pressure against overfitting)
obs_to_nnparams <- sim_params$n_obs / last(param_count)
tau0_scaling <- ifelse(
  (obs_to_nnparams > .5) | dont_scale_t0, 
  1, 
  obs_to_nnparams
) 

sim_params$prior_tau <- tau0_scaling * tau0_PV(
  p_0 = p_0, d = p+R, sig = 1, 
  n = sim_params$n_obs
)

agnostic_tau <- tau0_PV(
  p_0 = 1, d = 2, sig = 1, 
  n = sim_params$n_obs
)



# GEN DATA ----
# matching to description of synthetic data in sparseVCBART. 
# Experiment 1 uses p=3.  Experiment 2 uses p=50
# HOWEVER: function for \beta_1 described in paper appears to be wrong
# (doesn't even match their own plot).  The plot looks
# much closer to the function for \beta_2 in the orig. VCBART paper
# which is what I'm using instead.

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

set.seed(sim_params$seed)
Ey_df <- gen_Eydat_sparseVCBART(
  n_obs = round(n_obs / ttsplit), # training obs
  p = p,
  R = R,
  covar_fcn = corr_fcn,
  beta_0 = bfcns_list$beta_0,
  beta_1 = bfcns_list$beta_1,
  beta_2 = bfcns_list$beta_2,
  beta_3 = bfcns_list$beta_3
)
# generate epsilons
eps_mat <- matrix(
  rnorm(
    n = nrow(Ey_df),
    mean = mu_eps,
    sd = sig_eps
  ), 
  ncol = n_sims
)

## train/test ----
# Note: (test obs aren't used to calibrate NN,
# just for me to observe for training progress
# and catch simulation problems)
tr_inds <- 1:n_obs
te_inds <- (n_obs + 1):nrow(Ey_df)
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



# NETWORK SETUP ----
## define model
MLHS <- nn_module(
  "MLHS",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = sim_params$d_0, 
      out_features = sim_params$d_1,
      use_cuda = sim_params$use_cuda,
      tau_0 = sim_params$prior_tau,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc2 = torch_hs(
      in_features = sim_params$d_1,
      out_features = sim_params$d_2,
      use_cuda = sim_params$use_cuda,
      tau_0 = agnostic_tau,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc3 = torch_hs(
      in_features = sim_params$d_2,
      #   out_features = sim_params$d_3,
      #   use_cuda = sim_params$use_cuda,
      #   tau = agnostic_tau,
      #   init_weight = NULL,
      #   init_bias = NULL,
      #   init_alpha = 0.9,
      #   clip_var = TRUE
      # )
      # 
      # self$fc4 = torch_hs(
      #   in_features = sim_params$d_3,
      #   out_features = sim_params$d_4,
      #   use_cuda = sim_params$use_cuda,
      #   tau = agnostic_tau,
      #   init_weight = NULL,
      #   init_bias = NULL,
      #   init_alpha = 0.9,
      #   clip_var = TRUE
      # )
      # 
      # self$fc5 = torch_hs(
      #   in_features = sim_params$d_4,
      out_features = sim_params$d_L,
      use_cuda = sim_params$use_cuda,
      tau_0 = agnostic_tau,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
  },
  
  forward = function(x) {
    x %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      nnf_relu() %>%
      self$fc3() # %>%
    # nnf_relu() %>%
    # self$fc4() %>%
    # nnf_relu() %>%
    # self$fc5()
  },
  
  get_model_kld = function(){
    kl1 = self$fc1$get_kl()
    kl2 = self$fc2$get_kl()
    kl3 = self$fc3$get_kl()
    # kl4 = self$fc3$get_kl()
    # kl5 = self$fc3$get_kl()
    kld = kl1 + kl2 + kl3 #+ kl4 + kl5
    return(kld)
  }
)


# TRAIN LOOP ----






