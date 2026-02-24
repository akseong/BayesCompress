##################################################
## Project:   sparseVCBART sim - VANILLA BNN
## Date:      Feb 18, 2026
## Author:    Arnie Seong
##################################################


# VC structure **NOT** explicitly modeled.  
# Not super optimistic about performance here,
# at least in terms of recovering covariate functions

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

verbose <- TRUE



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
  "train_epochs" = 5E3,
  "report_every" = 1E2,
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


# DEFINE MODEL ----

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
Ey_raw <- Ey_df[, 1]
XZ_raw <- Ey_df[, -1]

# standardize XZ.  
# standardizing Y will have to happen after epsilons added
XZ_means <- colMeans(XZ_raw)
XZ_sds <- apply(XZ_raw, 2, sd)
XZ_centered <- sweep(XZ_raw, 2, STATS = XZ_means, "-")
XZ <- sweep(XZ_centered, 2, STATS = XZ_sds, "/")

XZ_tr <- torch_tensor(as.matrix(XZ[tr_inds, ]))
XZ_te <- torch_tensor(as.matrix(XZ[te_inds, ]))



# SIM_LOOP ----
######################################################
sim_ind <- 1
######################################################

## add noise and standardize y, test/train split ----
y_raw <- Ey_raw + eps_mat[, sim_ind]
y_mean <- mean(y_raw)
y_sd <- sd(y_raw)
y <- (y_raw - y_mean) / y_sd

y_tr <- torch_tensor(y[tr_inds])$unsqueeze(2)
y_te <- torch_tensor(y[te_inds])$unsqueeze(2)


if (sim_params$use_cuda){
  y_tr <- y_tr$to(device = "cuda")
  y_te <- y_te$to(device = "cuda")
  XZ_tr <- XZ_tr$to(device = "cuda")
  XZ_te <- XZ_te$to(device = "cuda")
}

## initialize BNN & optimizer ----
model_fit <- MLHS()
optim_model_fit <- optim_adam(model_fit$parameters, lr = sim_params$lr)

## store: # train, test mse and kl ----
report_epochs <- seq(
  sim_params$report_every, 
  sim_params$train_epochs, 
  by = sim_params$report_every
)

loss_mat <- matrix(
  NA, 
  nrow = length(report_epochs),
  ncol = 3
)
colnames(loss_mat) <- c("kl", "mse_train", "mse_test")
rownames(loss_mat) <- report_epochs


## store: alphas, kappas ----
alpha_mat <- matrix(
  NA, 
  nrow = length(report_epochs),
  ncol = sim_params$d_0
)
rownames(alpha_mat) <- report_epochs

kappa_local_mat <- 
  kappa_mat <- 
  kappa_tc_mat <-
  kappa_fc_mat <- alpha_mat


# TRAIN LOOP ----

# initialize training params
epoch <- 1
loss <- torch_tensor(1, device = dev_select(sim_params$use_cuda))

while (epoch <= sim_params$train_epochs){
  
  ## fit & metrics ----
  yhat_tr <- model_fit(XZ_tr)
  mse <- nnf_mse_loss(yhat_tr, y_tr)
  kl <- model_fit$get_model_kld() / length(y_tr)
  loss <- mse + kl
  
  # gradient step 
  # zero out previous gradients
  optim_model_fit$zero_grad()
  # backprop
  loss$backward()
  # update weights
  optim_model_fit$step()
  
  
  ## TRAIN PROGRESS ----
  time_to_report <- epoch!=0 & (epoch %% sim_params$report_every == 0)
  if (!time_to_report & verbose & (epoch %% sim_params$report_every == 1)){cat("Training till next report:")}
  if (!time_to_report & verbose & (epoch %% round(sim_params$report_every/100) == 1)){cat(".")}
  
  ### store results ----
  if (time_to_report){
    row_ind <- epoch %/% sim_params$report_every
    
    # compute test loss
    yhat_te <- model_fit(XZ_te)
    mse_te <- nnf_mse_loss(yhat_te, y_te)
    
    # store params
    loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_te$item())
    dropout_alphas <- model_fit$fc1$get_dropout_rates()
    alpha_mat[row_ind, ] <- as_array(dropout_alphas)
    kappas <- get_kappas(model_fit$fc1)
    kappa_mat[row_ind, ] <- kappas
    
    kappas_local <- get_kappas(model_fit$fc1, type = "local")
    kappa_local_mat[row_ind, ] <- kappas_local
    
    # corrected param kappas
    kappas_tc <- get_kappas_taucorrected(model_fit)
    kappas_fc <- get_kappas_frobcorrected(model_fit)
    kappa_fc_mat[row_ind, ] <- kappas_tc
    kappa_tc_mat[row_ind, ] <- kappas_fc
  }
  
  ### in-console reporting ----
  if (time_to_report & verbose){
    cat(
      "\n Epoch:", epoch,
      "MSE + KL/n =", round(mse$item(), 5), "+", round(kl$item(), 5),
      "=", round(loss$item(), 4),
      "\n",
      "train mse:", round(mse$item(), 4), 
      "; test_mse:", round(mse_te$item(), 4),
      sep = " "
    )
    
    # report global shrinkage params (s^2 or tau^2)
    s_sq1 <- get_s_sq(model_fit$fc1)
    s_sq2 <- get_s_sq(model_fit$fc2)
    s_sq3 <- get_s_sq(model_fit$fc3)
    
    cat(
      "\n s_sq1 = ", round(s_sq1, 5),
      "; s_sq2 = ", round(s_sq2, 5),
      "; s_sq3 = ", round(s_sq3, 5),
      sep = ""
    )
    # cat("\n alphas: ", round(as_array(dropout_alphas), 2), "\n")
    # display_alphas <- ifelse(
    #   as_array(dropout_alphas) <= sim_params$alpha_thresh,
    #   round(as_array(dropout_alphas), 3),
    #   "."
    # )
    # cat("alphas below 0.82: ")
    # cat(display_alphas, sep = " ")
    
    cat("\n global kappas: ", round(kappas, 2), "\n")
    cat("\n tau-corrected kappas: ", round(kappas_tc, 2), "\n")
    cat("\n frob-corrected kappas: ", round(kappas_fc, 2), "\n")
    cat(" \n \n")
  }
  
  # increment
  epoch <- epoch + 1
}






















