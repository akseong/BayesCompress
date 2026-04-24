##################################################
## Project:   meanfcns
## Date:      March 17, 2026
## Author:    Arnie Seong
##################################################

#  deeper nn, hshoe in first 2 layers only


#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe_smallbias.R"))
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))

# use cuda
if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}

# sim params and savepath ----
save_mod_path_prestem <- here::here(
  "sims", 
  "results", 
  "det1_smallbias_liangfcn1_"
)
n_obs <- 500 # includes training and test
d_in <- 100
sim_desc <- c(
  "Liang nonlin regression example, 
  5 MC samples for MSE, 
  kl annealing only - no lr annealing",
  "optimistic tau_0 (p_0 = 50 of 500)"
)

sim_params <- list(
  "sim_name" = sim_desc,
  # data params
  "n_obs" = n_obs,
  "d_in" = d_in,           ##
  "err_sig" = 1,          ##
  "ttsplit" = 4/5,        # Liang use 200 train, 300 test
  "genXfcn" = genX_mutualcorr,
  "meanfcn" = meanfcn_Liang1,
  "standardize" = TRUE,
  # sim params
  "seed" = 316,           ##
  "n_sims" = 2,           ##
  # network params / architecture
  "p_0frac" = 0.1,  ## expect about 1/10 covs to be included
  "d_1" = 16,
  "d_2" = 16,
  "d_3" = 16,
  "d_4" = 16,
  "d_5" = 16,
  "d_out" = 1,
  # training params
  "train_epochs" = 2e5,   
  "report_every" = 1E3,   
  "n_mc_samples" = 5,     
  "lr" = 0.001,  # If NULL, uses optim_adam default (0.001)
  "batch_size" = NULL,
  "lr_scheduler" = NULL, # torch::lr_cosine_annealing,
  "kl_scheduler" = kl_weight_cosine,
  "kl_warmup_frac" = 0.2,
  # don't usually modify
  "plot_every_x_reports" = 10,
  "use_cuda" = use_cuda
)
set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))

# calibrate tau ----
# Piironen & Vehtari 2017 suggest tau_0 = p_0 / (d - p_0) * sig / sqrt(n)
# where p_0 = prior estimate of number of nonzero betas, d = total number of covs

# scaled optimistic tau (expecting about 1/10 of covs to be included)
sim_params$prior_tau <- tau0_PV(
  p_0 = floor(sim_params$p_0frac * sim_params$d_in), 
  d = sim_params$d_in, 
  sig = 1,
  n = round(sim_params$n_obs * sim_params$ttsplit)
)

agnostic_tau <- tau0_PV(
  p_0 = 1, d = 2, sig = 1,
  n = round(sim_params$n_obs * sim_params$ttsplit)
)


## define model
MLHS <- nn_module(
  "MLHS",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = sim_params$d_in, 
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
    
    self$det1 = nn_linear(
      sim_params$d_2, 
    #   sim_params$d_3
    # )
    # 
    # self$det2 = nn_linear(
    #   sim_params$d_3, 
    #   sim_params$d_4
    # )
    # 
    # self$det3 = nn_linear(
    #   sim_params$d_4, 
    #   sim_params$d_5
    # )
    # 
    # self$det4 = nn_linear(
    #   sim_params$d_5, 
      sim_params$d_out
    )
    
    if (sim_params$use_cuda){
      self$det1$cuda()
      # self$det2$cuda()
      # self$det3$cuda()
      # self$det4$cuda()
    }
  },
  
  forward = function(x) {
    x %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      nnf_relu() %>%
      self$det1() # %>%
      # nnf_relu() %>%
      # self$det2() %>%
      # nnf_relu() %>%
      # self$det3() %>%
      # nnf_relu() %>%
      # self$det4() 
  },
  
  get_model_kld = function(){
    kl1 = self$fc1$get_kl()
    kl2 = self$fc2$get_kl()
    # kl3 = self$fc3$get_kl()
    # kl4 = self$fc4$get_kl()
    # kl5 = self$fc5$get_kl()
    kld = kl1 + kl2 # + kl3 + kl4 + kl5
    return(kld)
  }
)




res <- lapply(
  1:sim_params$n_sims,
  function(X) {
    save_mod_path_stem <- paste0(
      save_mod_path_prestem,
      sim_params$n_obs, "obs_", 
      sim_params$sim_seeds[X]
    )
    
    sim_hshoe_meanfcn(
      sim_ind = X,
      sim_params = sim_params,     # same as before, but need to include flist
      nn_model = MLHS,   # torch nn_module,
      verbose = TRUE,      # provide updates in console
      want_plots = FALSE,   # provide graphical updates of KL, MSE
      want_fcn_plots = FALSE, # display predicted functions
      save_fcn_plots = TRUE,
      want_all_params = TRUE,
      save_mod = TRUE,
      save_mod_path_stem = save_mod_path_stem
    )
  }
)

