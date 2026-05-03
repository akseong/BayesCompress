##################################################
## Project:   deeper nn, hshoe first layers only
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

# original function set
# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) - log(abs(x) + 1e-3)

# "smoother" functions (orig).  
fcn1 <- function(x) -cos(pi/1.5*x)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
fcn3 <- function(x) abs(x)^(.75)
fcn4 <- function(x) -x^2 / 4

# for new datagen:
# fcn1 <- function(x) -cos(pi/1.5*x)
# fcn2 <- function(x) sin(pi/1.5*x) + 1.5*cos(pi/1.2*(x-.5))
# fcn3 <- function(x) {
#   if (typeof(x)=="externalptr"){
#     (torch_round(x)%%2)*2 - 0.5
#   } else {
#     (round(x)%%2)*2 - 0.5  
#   }
# }
# fcn4 <- function(x) -x^2/4 + 2

# binary fcn
# fcn5 <- function(x) (x[, 1]^2)/4 - 2 + (x[, 2]<0)*3*cos(pi/2*x[, 1])

# x1 <- -30:30/10
# x2 <- sample(c(1, 0), length(x1), replace = TRUE)
# x2 <- 0
# y <- fcn5(cbind(x1, x2))
# plot(y~x1)

# # "smoother" functions, decentered
# fcn1 <- function(x) -sin(pi/1.5*x)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
# fcn3 <- function(x) abs(x)^(.75)
# fcn4 <- function(x) -x^2 / 4

flist = list(fcn1, fcn2, fcn3, fcn4)
# xlist <- list(1, 2, 3, 4, c(5, 6))
plot_datagen_fcns(flist)
# sim_func_data(n_obs, d_in, flist)

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
  "final_sims", 
  "results", 
  "nfdsmallbias_mutcorr0.5_5x16"
)

# can usually stop by 30k.  run to 50k epochs, anneal KL fract at .4
n_obs <- 500*10 # includes training and test

sim_desc <- c(
  "oldfcns, no minibatching, 5 MC samples for MSE, kl annealing only - no lr annealing",
  "optimistic tau_0 (p_0 = 10 of 104)"
)

sim_params <- list(
  "sim_name" = sim_desc,
  "n_obs" = n_obs,
  "err_sig" = 1,
  "xdist" = "norm",
  "xcorr" = NULL,
  "mut_corr" = 0.5,
  "xjitter" = NULL,
  "xshift" = NULL,
  "seed" = 516,
  "n_sims" = 10,
  "n_mc_samples" = 5,
  "train_epochs" = 5e4,
  "report_every" = 1E3,
  "plot_every_x_reports" = 10,
  "use_cuda" = use_cuda,
  "d_in" = 104,
  "d_hidden1" = 16,
  "d_hidden2" = 16,
  "d_hidden3" = 16,
  "d_hidden4" = 16,
  "d_hidden5" = 16,
  "d_out" = 1,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "alpha_thresh" = 1 / qchisq(1 - (0.05 / 104), df = 1),
  "flist" = flist,
  "lr" = 0.001,  # sim_hshoe learning rate arg.  If not specified, uses optim_adam default (0.001)
  "convergence_crit" = 1e-7,
  "ttsplit" = 4/5,
  "batch_size" = NULL,
  "stop_k" = 100,
  "stop_streak" = 25,
  "burn_in" = 25e4,
  "lr_scheduler" = NULL, # torch::lr_cosine_annealing,
  "kl_scheduler" = kl_weight_cosine,
  "kl_warmup_frac" = 0.4,
  "standardize" = TRUE
)
set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))

# corr_fcn <- function(i, j) {sim_params$xcorr^(abs(i-j))}
# sim_params$xcov <- make_Covmat(sim_params$d_in, corr_fcn)



# calibrate tau
# Piironen & Vehtari 2017 suggest tau_0 = p_0 / (d - p_0) * sig / sqrt(n)
# where p_0 = prior estimate of number of nonzero betas, d = total number of covs

# scaled optimistic tau (expecting 10 true of 104)
dim_vec <- do.call(c, sim_params[grep(pattern = "d_", names(sim_params))])
param_counts_from_dims(dim_vec)
param_scaling <- round(sim_params$n_obs * sim_params$ttsplit) / tail(param_counts_from_dims(dim_vec), 1)

sim_params$prior_tau <- tau0_PV(
  p_0 = 10, d = 104, sig = 1,
  n = round(sim_params$n_obs * sim_params$ttsplit)
)
# ) *
#   param_scaling

agnostic_tau <- tau0_PV(
  p_0 = 1, d = 2, sig = 1,
  n = round(sim_params$n_obs * sim_params$ttsplit)
)

# sim_params$prior_tau <- agnostic_tau <- 1


## define model
MLHS <- nn_module(
  "MLHS",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = sim_params$d_in, 
      out_features = sim_params$d_hidden1,
      use_cuda = sim_params$use_cuda,
      tau_0 = sim_params$prior_tau,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc2 = torch_hs(
      in_features = sim_params$d_hidden1,
      out_features = sim_params$d_hidden2,
      use_cuda = sim_params$use_cuda,
      tau_0 = agnostic_tau,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$det1 = nn_linear(
      sim_params$d_hidden2, 
      sim_params$d_hidden3
    )
    
    self$det2 = nn_linear(
      sim_params$d_hidden3, 
      sim_params$d_hidden4
    )
    
    self$det3 = nn_linear(
      sim_params$d_hidden4, 
      sim_params$d_hidden5
    )
    
    self$det4 = nn_linear(
      sim_params$d_hidden5, 
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
    
    sim_hshoe_det(
      sim_ind = X,
      sim_params = sim_params,     # same as before, but need to include flist
      nn_model = MLHS,   # torch nn_module,
      verbose = TRUE,      # provide updates in console
      want_plots = FALSE,   # provide graphical updates of KL, MSE
      want_fcn_plots = FALSE, # display predicted functions
      save_fcn_plots = FALSE,
      want_all_params = FALSE,
      save_mod = TRUE,
      save_mod_path_stem = save_mod_path_stem
    )
  }
)

