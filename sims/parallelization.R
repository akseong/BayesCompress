
#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))


#### GPU acceleration? ----
# not sure if will work on server....
if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}


#### data-generating functions ----
fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
fcn3 <- function(x) abs(x)^(1.5)
fcn4 <- function(x) - (abs(x))
flist = list(fcn1, fcn2, fcn3, fcn4)
# plot_datagen_fcns(flist)



#### sim_params ----
sim_params <- list(
  "sim_name" = "hot start, horseshoe, fcnal data",
  "seed" = 1002,
  "n_sims" = 2, 
  "train_epochs" = 1E5,
  "report_every" = 1E3,
  "use_cuda" = FALSE,    # use_cuda,
  "d_in" = 104,
  "d_hidden1" = 16,
  "d_hidden2" = 8,
  "d_out" = 1,
  "n_obs" = 125000, # more n_obs --> fewer training epochs needed
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "wald_thresh" = 1 / qchisq(1 - (0.05 / 104), df = 1),
  "flist" = flist,
  "err_sig" = 1,
  "convergence_crit" = 1e-7,
  "ttsplit" = 4/5,
  "stop_k" = 100,
  "stop_streak" = 25,
  "burn_in" = 2E5
)

save_fname <- paste0(
  "hshoe_fcnl_multi",
  sim_params$n_obs,
  "_maxepochs",
  sim_params$seed,
  "_TESTING",
  ".RData"
)

set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))

# # hot_start_simulate data with set.seed and torch set seed ----
# sim_ind <- 1   # comment out when not testing
# sim_params$hot_start_epochs <- 5E5
# set.seed(sim_params$sim_seeds[sim_ind])
# torch_manual_seed(sim_params$sim_seeds[sim_ind])
# 
# DNN <- hot_start_DNN(
#     sim_ind = 1,
#     sim_params,
#     save_mod = FALSE,
#     save_path = NULL
# )


## define model
MLHS <- nn_module(
  "MLHS",
  initialize = function() {
    self$fc1 = torch_hs(
      in_features = sim_params$d_in,
      out_features = sim_params$d_hidden1,
      use_cuda = FALSE,
      tau = 1,
      init_weight = NULL, # DNN$parameters$`0.weight`,
      init_bias = NULL,   # DNN$parameters$`0.bias`,
      init_alpha = 0.9,
      clip_var = TRUE
    )

    self$fc2 = torch_hs(
      in_features = sim_params$d_hidden1,
      out_features = sim_params$d_hidden2,
      use_cuda = FALSE,
      tau = 1,
      init_weight = NULL, #DNN$parameters$`2.weight`,
      init_bias = NULL,   #DNN$parameters$`2.bias`,
      init_alpha = 0.9,
      clip_var = TRUE
    )

    self$fc3 = torch_hs(
      in_features = sim_params$d_hidden2,
      out_features = sim_params$d_out,
      use_cuda = FALSE,
      tau = 1,
      init_weight = NULL, #DNN$parameters$`4.weight`,
      init_bias = NULL,   #DNN$parameters$`4.bias`,
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
      self$fc3()
  },



  get_model_kld = function(){
    kl1 = self$fc1$get_kl()
    kl2 = self$fc2$get_kl()
    kl3 = self$fc3$get_kl()
    kld = kl1 + kl2 + kl3
    return(kld)
  }
)



res <- sim_fcn_hshoe_fcnaldata(
  sim_ind = 1,
  sim_params = sim_params,
  nn_model = MLHS,
  train_epochs = 1E6, # sim_params$train_epochs,
  verbose = TRUE,
  display_alpha_thresh = sim_params$wald_thresh,
  report_every = 1E3, # sim_params$report_every,
  want_plots = FALSE,
  want_fcn_plots = TRUE,
  save_mod = TRUE,
  stop_k = 100,
  stop_streak = 25,
  burn_in = 5E5
)



# res <- lapply(
#   1:sim_params$n_sims, 
#   function(X) sim_fcn_hshoe_fcnaldata(
#     sim_ind = X, 
#     sim_params = sim_params,
#     nn_model = MLHS,
#     train_epochs = sim_params$train_epochs,
#     verbose = TRUE,
#     display_alpha_thresh = sim_params$wald_thresh,
#     report_every = sim_params$report_every,
#     want_plots = FALSE,
#     want_fcn_plots = TRUE,
#     save_mod = TRUE
#   )
# )
# 
# # set each simulation result unique name
# res <- setNames(res, paste0("sim_", 1:length(res)))
# 
# 
# contents <- list(
#   "res" = res, 
#   "sim_params" = sim_params
# )
# 
# save(contents, file = here::here("sims", "results", save_fname))
# 
# 














