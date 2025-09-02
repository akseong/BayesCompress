
#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))


#### regression model ----

## sim_params
#    check whenever changing setting (testing / single vs parallel, etc) ##
#           n_sims, verbose, want_plots, train_epochs
sim_params <- list(
  "sim_name" = "horseshoe, linear regression setting, KL scaled by n",
  "n_sims" = 10, 
  "d_in" = 104,
  "d_hidden1" = 16,
  "d_hidden2" = 16,
  "d_out" = 1,
  "n_obs" = 1250,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "seed" = 5,
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
set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))


## define model
MLHS <- nn_module(
  "MLHS",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = sim_params$d_in, 
      out_features = sim_params$d_hidden1,
      use_cuda = FALSE,
      tau = 1,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc2 = torch_hs(
      in_features = sim_params$d_hidden1,
      out_features = sim_params$d_hidden2,
      use_cuda = FALSE,
      tau = 1,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc3 = torch_hs(
      in_features = sim_params$d_hidden2,
      out_features = sim_params$d_out,
      use_cuda = FALSE,
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



# sim_fcn_hshoe_linreg() is in sim_functions.R
res <- lapply(
  1:sim_params$n_sims, 
  function(X) sim_fcn_hshoe_linreg(
    sim_ind = X, 
    sim_params = sim_params,
    nn_model = MLHS,
    train_epochs = 250000,
    verbose = FALSE,
    report_every = 1000,
    want_plots = FALSE,
    want_all_params = FALSE,
    want_data = FALSE
  )
)

# set each simulation result unique name
res <- setNames(res, paste0("sim_", 1:length(res)))


contents <- list(
  "res" = res, 
  "sim_params" = sim_params
)
save(contents, file = here::here("sims", "results", "hshoe_overparam1000_maxepochs5.RData"))
















