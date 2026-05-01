##################################################
## Project:   simulations - original functions, 0.5 mutual corr
## Date:      May 1, 2026
## Author:    Arnie Seong
##################################################


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
data_seed <- 516
n_obs <- 1000 # includes training and test
d_in <- 20
n_nets <- 5
n_sims <- 5

save_mod_path_prestem <- here::here(
  "final_sims", 
  "results", 
  paste0(
    "test_orig_mcorr.5_p20_",
    n_obs, "_dataseed", data_seed
  )
)

sim_desc <- c(
  "original meanfcns nonlin regression example, 
  P=50, train obs = 800, 200 test
  5 MC samples for MSE, 
  kl annealing only - no lr annealing",
  "optimistic tau_0 set so p_0 = p/20"
)


orig_fcns <- function(x, round_dig = NULL){
  -cos(pi/1.5*x[,1]) + cos(pi*x[,2]) + sin(pi/1.2*x[,2])
  + abs(x[,3])^(.75) - x[,4]^2/4
}


sim_params <- list(
  "sim_name" = sim_desc,
  # data params
  "n_obs" = n_obs,
  "d_in" = d_in,           ##
  "err_sig" = 1,          ##
  "mut_corr" = 0.5,
  "ttsplit" = 4/5,        # Liang use 200 train, 300 test
  "genXfcn" = genX_mutualcorr,
  "meanfcn" = orig_fcns,
  "standardize" = TRUE,
  # sim params
  "data_seed" = data_seed,           ##
  "n_nets" = n_nets,           ##
  # network params / architecture
  "p_0frac" = 0.2,  ## optimistic; expect about 1/5 covs to be included
  "d_1" = 16,
  "d_2" = 16,
  "d_3" = 16,
  "d_4" = 16,
  "d_5" = 16,
  "d_out" = 1,
  # training params
  "train_epochs" = 2e5,   
  "report_every" = 1E3,
  "save_every" = 25e3,
  "save_after" = 75e3,
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
set.seed(sim_params$data_seed)
sim_params$nn_init_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))

# calibrate tau ----
# Piironen & Vehtari 2017 suggest tau_0 = p_0 / (d - p_0) * sig / sqrt(n)
# where p_0 = prior estimate of number of nonzero betas, d = total number of covs

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


## define network ----
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
      sim_params$d_3
    )
    
    self$det2 = nn_linear(
      sim_params$d_3,
      sim_params$d_4
    )
    
    self$det3 = nn_linear(
      sim_params$d_4,
      sim_params$d_5
    )
    
    self$det4 = nn_linear(
      sim_params$d_5,
      sim_params$d_out
    )
    
    if (sim_params$use_cuda){
      self$det1$cuda()
      self$det2$cuda()
      self$det3$cuda()
      self$det4$cuda()
    }
  },
  
  forward = function(x) {
    x %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      nnf_relu() %>%
      self$det1() %>%
      nnf_relu() %>%
      self$det2() %>%
      nnf_relu() %>%
      self$det3() %>%
      nnf_relu() %>%
      self$det4()
  },
  
  get_model_kld = function(){
    kl1 = self$fc1$get_kl()
    kl2 = self$fc2$get_kl()
    kld = kl1 + kl2 
    return(kld)
  }
)



# GEN DATA ----
set.seed(sim_params$data_seed)
simdat <- sim_meanfcn_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    mut_corr = sim_params$mut_corr,
    genXfcn = sim_params$genXfcn,
    meanfcn = sim_params$meanfcn,
    err_sigma = sim_params$err_sig,
    round_dig = NULL,
    standardize = FALSE
)
sim_params$simdat <- simdat

if (sim_params$use_cuda){
  simdat$x <- simdat$x$to(device = "cuda")
  simdat$y <- simdat$y$to(device = "cuda")
}
split_simdat <- ttsplit_simdat(simdat, ttsplit_ratio = sim_params$ttsplit)

# standardize ----
x_mean <- torch_mean(split_simdat$x_train, dim = 1, keepdim = TRUE)
x_sd <- torch_std(split_simdat$x_train, dim = 1, keepdim = TRUE)
y_mean <- torch_mean(split_simdat$y_train, dim = 1, keepdim = TRUE)
y_sd <- torch_std(split_simdat$y_train, dim = 1, keepdim = TRUE)

x_train <- (split_simdat$x_train - x_mean)/x_sd
y_train <- (split_simdat$y_train - y_mean)/y_sd
x_test <- (split_simdat$x_test - x_mean)/x_sd
y_test <- (split_simdat$y_test - y_mean)/y_sd

sim_params$x_mean <- x_mean
sim_params$x_sd <- x_sd
sim_params$y_mean <- y_mean
sim_params$y_sd <- y_sd
sim_params$train_sig <- sim_params$err_sig / y_sd$item()

## estimating snr / track train progress
sim_params$train_sig <- sim_params$err_sig / y_sd$item()

# train multiple networks----
save_net_path_stem <- paste0(
  save_mod_path_prestem,
  "_net"
)
sim_params$net_path_stem <- save_net_path_stem
cat_color(paste0("mse target: ", round(sim_params$train_sig, 4), "\n"))

res_all <- lapply(
  1:sim_params$n_nets,
  function(net_ind){
    sim_hshoe_simdat(
      net_ind = net_ind,
      sim_params = sim_params,
      x_train = x_train,
      x_test = x_test,
      y_train = y_train,
      y_test = y_test,
      nn_model = MLHS,
      verbose = TRUE
    )
  }
)

save(res_all, file = paste0(save_mod_path_prestem, "_all.RData"))







# res <- lapply(
#   1:sim_params$n_sims,
#   function(X) {
#     save_mod_path_stem <- paste0(
#       save_mod_path_prestem,
#       sim_params$n_obs, "obs_", 
#       sim_params$sim_seeds[X]
#     )
#     
#     
#     
#     sim_hshoe_meanfcn(
#       sim_ind = X,
#       sim_params = sim_params,     # same as before, but need to include flist
#       nn_model = MLHS,   # torch nn_module,
#       verbose = TRUE,      # provide updates in console
#       want_plots = FALSE,   # provide graphical updates of KL, MSE
#       want_fcn_plots = FALSE, # display predicted functions
#       save_fcn_plots = TRUE,
#       want_all_params = TRUE,
#       save_mod = TRUE,
#       save_mod_path_stem = save_mod_path_stem
#     )
#   }
# )
# 
# 
