##################################################
## Project:   retrain / continue training
## Date:      Sep 24, 2025
## Author:    Arnie Seong
##################################################


#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))


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
fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x
fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) -x^2 / 2 -3
fcn4 <- function(x) - log(abs(x) + 1e-3)
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

sim_params <- list(
  "sim_name" = "hshoe, tau fixed, 2 layers 16 8 (worked before) nobatching, fcnal data.  ",
  "seed" = 2168,
  "n_sims" = 1, 
  "train_epochs" = 5e5, # 15E5,
  "report_every" = 1e4, # 1E4,
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

nn_model <- torch::torch_load(here::here("sims", "results", "fcnl_hshoe_mod_12500obs_398060.pt"))

if (sim_params$use_cuda){
  nn_model <- nn_model$to(device = "cuda")
} else {
  nn_model <- nn_model$to(device = "cpu")
}



sim_continue_training(
    sim_seed = 398060,
    sim_params = sim_params,     # same as before, but need to include flist
    nn_model = nn_model,
    verbose = TRUE,   # provide updates in console
    want_plots = TRUE,   # provide graphical updates of KL, MSE
    want_fcn_plots = TRUE,   # display predicted functions
    save_fcn_plots = TRUE,
    want_all_params = FALSE,
    save_mod = TRUE,
    save_results = TRUE,
    save_mod_path_stem = NULL
)

