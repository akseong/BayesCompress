##################################################
## Project:   retrain reduced mod
## Date:      Oct 22, 2025
## Author:    Arnie Seong
##################################################

# setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))


# load
seeds <- c(966694, 191578, 272393, 718069, 377047)
mod_stem <- here::here("sims", "results", "fcnl_hshoe_mod_12500obs_")
mod_fnames <- paste0(mod_stem, seeds, ".pt")
res_fnames <- paste0(mod_stem, seeds, ".RData")
seednum <- 2
nn_model <- torch::torch_load(mod_fnames[seednum], device = "cpu")
load(res_fnames[seednum])


# RETRAIN ----
sim_res$loss_mat
sim_params <- sim_res$sim_params
if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}
sim_params$use_cuda <- use_cuda

set.seed(seeds[seednum])
torch_manual_seed(seeds[seednum])
simdat <- sim_func_data(
  n_obs = sim_params$n_obs, 
  d_in = sim_params$d_in, 
  flist = sim_params$flist, 
  err_sigma = sim_params$err_sig,
)
if (nn_model$fc1$atilde_logvar$is_cuda){
  simdat$x <- simdat$x$to(device = "cuda")
  simdat$y <- simdat$x$to(device = "cuda")
}

## visualize ----
pred_mats <- make_pred_mats(
  flist = sim_params$flist,
  d_in = sim_params$d_in
)

fcn_plt <- plot_datagen_fcns(
  flist = sim_params$flist,
  min_x = -5,
  max_x = 5
)

plot_fcn_preds(torchmod = nn_model, pred_mats)


## selecting nodes with dropout probability < mean (mean-ish model?) ----
k1 <- get_kappas(nn_model$fc1)
round(k1, 3)

k2 <- get_kappas(nn_model$fc2)
round(k2, 3)
l1_selected_nodes <- k2 < mean(k2)
d1_red <- sum(l1_selected_nodes)
d1_red

k3 <- get_kappas(nn_model$fc3)
round(k3, 3)  
l2_selected_nodes <- k3 < 1
d2_red <- sum(l2_selected_nodes)
d2_red


# define model, initialize params:
## define model
MLHS_red <- nn_module(
  "MLHS_red",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = sim_params$d_in, 
      out_features = d1_red,
      use_cuda = sim_params$use_cuda,
      tau = 1,
      init_weight = nn_model$fc1$weight_mu[l1_selected_nodes, ],
      init_bias = nn_model$fc1$bias_mu[l1_selected_nodes],
      init_sa = nn_model$fc1$sa_mu, 
      init_sb = nn_model$fc1$sb_mu, 
      init_atilde = nn_model$fc1$atilde_mu, 
      init_btilde = nn_model$fc1$btilde_mu, 
      init_weight_logvar = nn_model$fc1$weight_logvar[l1_selected_nodes, ], 
      init_bias_logvar = nn_model$fc1$bias_logvar[l1_selected_nodes], 
      init_sa_logvar = nn_model$fc1$sa_logvar, 
      init_sb_logvar = nn_model$fc1$sb_logvar, 
      init_atilde_logvar = nn_model$fc1$atilde_logvar, 
      init_btilde_logvar = nn_model$fc1$btilde_logvar, 
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc2 = torch_hs(
      in_features = sim_params$d_hidden1,
      out_features = d2_red,
      use_cuda = sim_params$use_cuda,
      tau = 1,
      init_weight = nn_model$fc2$weight_mu[l2_selected_nodes, l1_selected_nodes],
      init_bias = nn_model$fc2$bias_mu[l2_selected_nodes],
      init_sa = nn_model$fc2$sa_mu, 
      init_sb = nn_model$fc2$sb_mu, 
      init_atilde = nn_model$fc2$atilde_mu[l1_selected_nodes], 
      init_btilde = nn_model$fc2$btilde_mu[l1_selected_nodes], 
      init_weight_logvar = nn_model$fc2$weight_logvar[l2_selected_nodes, l1_selected_nodes], 
      init_bias_logvar = nn_model$fc2$bias_logvar[l2_selected_nodes], 
      init_sa_logvar = nn_model$fc2$sa_logvar, 
      init_sb_logvar = nn_model$fc2$sb_logvar, 
      init_atilde_logvar = nn_model$fc2$atilde_logvar[l1_selected_nodes], 
      init_btilde_logvar = nn_model$fc2$btilde_logvar[l1_selected_nodes], 
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc3 = torch_hs(
      in_features = d2_red,
      #   out_features = sim_params$d_hidden3,
      #   use_cuda = sim_params$use_cuda,
      #   tau = 1,
      #   init_weight = NULL,
      #   init_bias = NULL,
      #   init_alpha = 0.9,
      #   clip_var = TRUE
      # )
      # 
      # self$fc4 = torch_hs(
      #   in_features = sim_params$d_hidden3,
      #   out_features = sim_params$d_hidden4,
      #   use_cuda = sim_params$use_cuda,
      #   tau = 1,
      #   init_weight = NULL,
      #   init_bias = NULL,
      #   init_alpha = 0.9,
      #   clip_var = TRUE
      # )
      # 
      # self$fc5 = torch_hs(
      #   in_features = sim_params$d_hidden4,
      out_features = sim_params$d_out,
      use_cuda = sim_params$use_cuda,
      tau = 1,
      init_weight = nn_model$fc3$weight_mu[, l2_selected_nodes],
      init_bias = nn_model$fc3$bias_mu,
      init_sa = nn_model$fc3$sa_mu, 
      init_sb = nn_model$fc3$sb_mu, 
      init_atilde = nn_model$fc3$atilde_mu[l2_selected_nodes], 
      init_btilde = nn_model$fc3$btilde_mu[l2_selected_nodes], 
      init_weight_logvar = nn_model$fc3$weight_logvar[, l2_selected_nodes], 
      init_bias_logvar = nn_model$fc3$bias_logvar, 
      init_sa_logvar = nn_model$fc3$sa_logvar, 
      init_sb_logvar = nn_model$fc3$sb_logvar, 
      init_atilde_logvar = nn_model$fc3$atilde_logvar[l2_selected_nodes], 
      init_btilde_logvar = nn_model$fc3$btilde_logvar[l2_selected_nodes], 
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

sim_params$model_red <- MLHS_red
sim_params$sim_name <- paste0("reduced layer 1 only based on kappas from: ", sim_params$sim_name)
sim_params$d_hidden1 <- d1_red
sim_params$d_hidden2 <- d2_red
save_mod_path_stem <- here::here("sims", 
                                 "results", 
                                 paste0("fcnl_hshoe_mod_", 
                                        sim_params$n_obs, "obs_",
                                        seeds[seednum],
                                        "_REDL1only"
                                 ))

sim_params$train_epochs <- 5e5
sim_params$report_every <- 10000

sim_res_red <- sim_hshoe(
  seed = seeds[seednum],
  sim_ind = NULL,
  sim_params,     # same as before, but need to include flist
  nn_model = MLHS_red,   # torch nn_module,
  verbose = TRUE,   # provide updates in console
  want_plots = FALSE,   # provide graphical updates of KL, MSE
  want_fcn_plots = TRUE,   # display predicted functions
  save_fcn_plots = TRUE,
  want_all_params = TRUE,
  local_only = FALSE,
  save_mod = TRUE,
  save_results = TRUE,
  save_mod_path_stem = save_mod_path_stem
)










