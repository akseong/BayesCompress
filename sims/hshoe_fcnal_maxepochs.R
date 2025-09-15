
#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe_cuda.R"))
source(here("Rcode", "sim_functions.R"))


if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}



fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
fcn3 <- function(x) abs(x)^(1.5)
fcn4 <- function(x) - (abs(x))
flist = list(fcn1, fcn2, fcn3, fcn4)
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
  "sim_name" = "horseshoe, fcnal data",
  "seed" = 1002,
  "n_sims" = 2, 
  "train_epochs" = 75E4,
  "report_every" = 1E3,
  "use_cuda" = use_cuda,
  "d_in" = 104,
  "d_hidden1" = 16,
  "d_hidden2" = 8,
  "d_out" = 1,
  "n_obs" = 12500,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "wald_thresh" = 1 / qchisq(1 - (0.05 / 104), df = 1),
  "flist" = flist,
  "err_sig" = 1,
  "burn_in" = 1,
  "convergence_crit" = 1e-7,
  "ttsplit" = 4/5,
  "stop_criteria_interval" = 50,
  "moving_avg_interval" = 17,
  "stop_criteria" = c(
    "test_train",        # [te_loss - tr_loss] positive & increasing for [stop_criteria_interval] epochs
    "train_convergence", # tr_loss_diff < [convergence_crit] for [stop_cruit_interval] epochs
    "test_convergence",  # te_loss_diff < [convergence_crit] for ...
    "ma_loss_increasing" # ma_tr_loss increasing for ...
  )
)
save_fname <- paste0(
  "hshoe_fcnldata",
  sim_params$n_obs,
  "_maxepochs",
  sim_params$seed,
  ".RData"
)

set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))


## define model
MLHS <- nn_module(
  "MLHS",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = sim_params$d_in, 
      out_features = sim_params$d_hidden1,
      use_cuda = sim_params$use_cuda,
      tau = 1,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc2 = torch_hs(
      in_features = sim_params$d_hidden1,
      out_features = sim_params$d_hidden2,
      use_cuda = sim_params$use_cuda,
      tau = 1,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc3 = torch_hs(
      in_features = sim_params$d_hidden2,
      out_features = sim_params$d_out,
      use_cuda = sim_params$use_cuda,
      tau = 1,
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


source(here("scratch_code.R"))
test <- sim_fcn_hshoe_fcnaldata(
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
  burn_in = 5E4
)




# sim_fcn_hshoe_linreg() is in sim_functions.R
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














