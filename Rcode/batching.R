
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
  "train_epochs" = 1e6, # 15E5,
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

sim_params$model <- MLHS
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
  function(X) sim_hshoe(
    sim_ind = X,
    sim_params = sim_params,     # same as before, but need to include flist
    nn_model = MLHS,   # torch nn_module,
    verbose = TRUE,      # provide updates in console
    want_plots = FALSE,   # provide graphical updates of KL, MSE
    want_fcn_plots = TRUE, # display predicted functions
    save_fcn_plots = TRUE,
    want_all_params = FALSE,
    save_mod = TRUE,
    save_mod_path_stem = NULL
  )
)

# run 1, on CPU: , batch size 2^5 (32)
# 10k  11:21, ~ 14 min; train_mse: 194233.7 ; test_mse: 2783824 
# 20k  11:31, 10 min; train mse: 1362.434 ; test_mse: 1182.7 
# 30k

# no batching, on CPU, d1 = 16, d2 = 8, tau no longer learnable


