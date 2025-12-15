##################################################
## Project:   Kaiming init, smoother functions
## Date:      Dec 9, 2025
## Author:    Arnie Seong
##################################################

# Kaiming initialization used
# normalize train data

#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe_klcorrected.R"))
source(here("Rcode", "sim_functions.R"))
# source(here("Rcode", "sim_hshoe_normedresponse.R"))

if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}

# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) - (abs(x))
# flist = list(fcn1, fcn2, fcn3, fcn4)
# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) -x^2 / 2 -3
# fcn4 <- function(x) - log(abs(x) + 1e-3)
fcn1 <- function(x) -cos(pi/1.5*x)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
fcn3 <- function(x) abs(x)^(.75)
fcn4 <- function(x) -x^2 / 4
flist = list(fcn1, fcn2, fcn3, fcn4)
plot_datagen_fcns(flist)
# 
# xshow <- seq(-3, 3, length.out = 100)
# yshow <- sapply(flist, function(fcn) fcn(xshow))
# df <- data.frame(
#   "f1" = yshow[, 1],
#   "f2" = yshow[, 2],
#   "f3" = yshow[, 3],
#   "f4" = yshow[, 4],
#   "x"  = xshow
# )
# df %>% 
#   pivot_longer(cols = -x, names_to = "fcn") %>% 
#   ggplot(aes(y = value, x = x, color = fcn)) +
#   geom_line() + 
#   labs(title = "functions used to create data")

## sim_params
#    check whenever changing setting (testing / single vs parallel, etc) ##
#           n_sims, verbose, want_plots, train_epochs


save_mod_path_prestem <- here::here(
  "sims", 
  "results", 
  "hshoe_smooth_kaiming_"
)

sim_params <- list(
  "sim_name" = "tau_0 = 1, kaiming init, 2 layers 16 8, nobatching, fcnal data.  ",
  "seed" = 21683,
  "n_sims" = 1, 
  "train_epochs" = 15E5,
  "report_every" = 1E4,
  "use_cuda" = use_cuda,
  "d_in" = 104,
  "d_hidden1" = 16,
  "d_hidden2" = 8,
  # "d_hidden3" = 8,
  "d_out" = 1,
  "n_obs" = 12500,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "alpha_thresh" = 1 / qchisq(1 - (0.05 / 104), df = 1),
  "flist" = flist,
  "lr" = 0.05,
  "err_sig" = 1,
  "convergence_crit" = 1e-7,
  "ttsplit" = 4/5,
  "batch_size" = NULL,
  "stop_k" = 100,
  "stop_streak" = 25,
  "burn_in" = 25e4 # 5E5,
)
set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))

# Piironen & Vehtari 2017 suggest tau_0 = p_0 / (d - p_0) * sig / sqrt(n)
# where p_0 = prior estimate of number of nonzero betas, d = total number of covs
tau0_PV <- function(p_0, d, sig = 1, n){
  p_0 / (d - p_0) * sig / sqrt(n)
}

prior_tau <- tau0_PV(p_0 = 50, d = 100, sig = 1, n = 1e4)

## define model
MLHS <- nn_module(
  "MLHS",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = sim_params$d_in, 
      out_features = sim_params$d_hidden1,
      use_cuda = sim_params$use_cuda,
      tau_0 = 1,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc2 = torch_hs(
      in_features = sim_params$d_hidden1,
      out_features = sim_params$d_hidden2,
      use_cuda = sim_params$use_cuda,
      tau_0 = 1,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc3 = torch_hs(
      in_features = sim_params$d_hidden2,
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
      tau_0 = 1,
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
# 
# sim_params$model <- MLHS
# mod <- MLHS()
# mod$fc1$parameters
# ztil <- get_ztil_sq(mod$fc1)
# s <- get_s_sq(mod$fc1)
# Wz_params <- get_Wz_params(mod$fc1)
# Wz_mu <- Wz_params$Wz_mu
# W_mu <- as_array(mod$fc1$weight_mu)
# mean(diag(cov(W_mu)))
# 
# 
# nn_init_kaiming_normal_(mod$fc1$weight_mu)
# W_mu_k <- as_array(mod$fc1$weight_mu)
# sum(diag(cov(W_mu_k)))
# sum(diag(cov(W_mu)))
# mean(diag(cov(W_mu_k)))
# mean(W_mu_k)
# mean(W_mu)

# verbose = TRUE
# want_plots = TRUE
# want_fcn_plots = TRUE
# save_fcn_plots = FALSE
# want_all_params = FALSE
# save_mod = TRUE
# save_mod_path_stem = NULL
# nn_model <- MLHS

res <- lapply(
  1:sim_params$n_sims,
  function(X) {
    save_mod_path_stem <- paste0(
      save_mod_path_prestem,
      sim_params$n_obs, "obs_", 
      sim_params$sim_seeds[X]
    )
    
    sim_hshoe(
      sim_ind = X,
      sim_params = sim_params,     # same as before, but need to include flist
      nn_model = MLHS,   # torch nn_module,
      verbose = TRUE,      # provide updates in console
      want_plots = FALSE,   # provide graphical updates of KL, MSE
      want_fcn_plots = TRUE, # display predicted functions
      save_fcn_plots = TRUE,
      want_all_params = TRUE,
      save_mod = TRUE,
      save_mod_path_stem = save_mod_path_stem
    )
  }
)

