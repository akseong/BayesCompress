##################################################
## Project:   HORSESHOE MODEL simulation code
## Date:      Jul 11, 2025
## Author:    Arnie Seong
##################################################

#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)


library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))

`%notin%` <- Negate(`%in%`)





#### linear_sim ----
## params
params <- list(
  "d_in" = 104,
  "n_obs" = 10000,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "seed" = 314,
  "err_sig" = 1,
  "verbose" = TRUE,
  "report_every" = 1000, # training epochs between display/store results
  "want_plots" = TRUE,   # set to FALSE when running in parallel
  "train_epochs" = 200000,
  "burn_in" = 5000,
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


## define model
SLHS <- nn_module(
  "SLHS",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = params$d_in, 
      out_features = 1,
      use_cuda = FALSE,
      tau = 1,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = NULL
    )
  },
  
  forward = function(x) {
    x %>% 
      self$fc1() 
  },
  
  get_model_kld = function(){
    kld = self$fc1$get_kl()
    return(kld)
  }
)


## generate linear data
set.seed(params$seed)
torch_manual_seed(params$seed)
slhs_net <- SLHS()
lin_simdat <- sim_linear_data(
  n = params$n_obs,
  true_coefs = params$true_coefs,
  err_sigma = params$err_sig
)

# store: # train, test loss
report_epochs <- seq(
  params$report_every, 
  params$train_epochs, 
  by = params$report_every
)
loss_mat <- matrix(
  NA, 
  nrow = length(report_epochs),
  ncol = 6
)
colnames(loss_mat) <- c(
  "tr_loss", "te_loss", 
  "tr_kl", "te_kl", 
  "tr_mse", "te_mse"
)
rownames(loss_mat) <- report_epochs

# store: alphas
alpha_mat <- matrix(
  NA, 
  nrow = length(report_epochs),
  ncol = params$d_in
)
rownames(alpha_mat) <- report_epochs



## initialize training params ----
loss_diff <- 1
loss <- torch_zeros(1)
epoch <- 0
stop_criteria_met <- FALSE


## SETUP STOP CRITERIA ----
# store: moving avgs (for stop criteria)
crit_raw_mat <- matrix(NA, ncol = 4, nrow = 2 * params$moving_avg_interval)
crit_ma_mat <- matrix(NA, ncol = 4, nrow = 2 * params$stop_criteria_interval)
colnames(crit_raw_mat) <- c("test_train_diff", "loss_diff", "loss")
colnames(crit_ma_mat) <- c("test_train_diff", "loss_diff", "loss")


## test-train split
ttsplit_used <- "test_train" %in% params$stop_criteria || "test_convergence" %in% params$stop_criteria
ttsplit_ind <- ifelse(
  ttsplit_used,
  floor(params$n_obs * params$ttsplit),
  params$n_obs
)

if (ttsplit_used){
  x_test <- lin_simdat$x[(ttsplit_ind+1):params$n_obs, ] 
  y_test <- lin_simdat$y[(ttsplit_ind+1):params$n_obs, ]
  loss_test <- torch_zeros(1)
  loss_diff_test <- 1
}

x_train <- lin_simdat$x[1:ttsplit_ind, ]
y_train <- lin_simdat$y[1:ttsplit_ind, ]

## optimizer
optim_slhs <- optim_adam(slhs_net$parameters)

## TRAINING LOOP ----
# while (!stop_criteria_met){
  prev_loss <- loss
  epoch <- epoch + 1
  
  yhat_train <- slhs_net(x_train)
  mse <- nnf_mse_loss(yhat_train, y_train)
  kl <- slhs_net$get_model_kld() / ttsplit_ind

  loss <- mse + kl
  loss_diff <- (loss - prev_loss)$item()

  # gradient step 
  # zero out previous gradients
  optim_slhs$zero_grad()
  # backprop
  loss$backward()
  # update weights
  optim_slhs$step()

  
  # compute test loss 
  if (ttsplit_used) {
    prev_loss_test <- loss_test
    yhat_test <- slhs_net(x_test) 
    # **WOULD LIKE THIS TO BE DETERMINISTIC, i.e. based on post pred means** ----
    mse_test <- nnf_mse_loss(yhat_test, y_test)
    kl_test <- slhs_net$get_model_kld() / (lin_simdat$n - ttsplit_ind)
    # THESE KLS ARE TOO DIFFERENT TO BE COMPARABLE??
    loss_test <- mse_test + kl_test
    loss_diff_test <- (loss_test - prev_loss_test)$item()
  }
  

  # should stop criteria on test set be on just MSE?  or also KL?
  # check what happens if not scaling KL by number of obs used to compute
  # should really be scaled by number of params (if anything at all), no?  
  #  seems like the real reason KL is typically scaled is for BATCH gradient descent
  # Figure out how to get yhats BASED ON POSTERIOR PREDITIVE MEANS
  # MAYBE START 
  
  
  # CHECK STOP CRITERIA ----
  
  if ("convergence" %in% params$stop_criteria){
    
    
    
    
  }
  
  
  

  if(epoch %% 1000 == 0 & verbose){
    cat(
      "Epoch: ", epoch,
      "MSE + KL/n = ", mse$item(), " + ", kl$item(),
      " = ", loss$item(),
      "\n"
    )
    # Eztilde <- slhs_net$fc1$get_Eztilde_i()
    # Vztilde <- slhs_net$fc1$get_Vztilde_i()
    # cat("E[ztilde_i]: ", round(as_array(Eztilde), 4), "\n \n")
    # cat("V[ztilde_i]: ", round(as_array(Vztilde), 4), "\n \n \n")
    # cat(round(as_array(slnj_net$fc1$z_mu), 4), "\n")
    w_mu <- slhs_net$fc1$compute_posterior_param()$post_weight_mu
    cat("E[W|y]: ", round(as_array(w_mu), 2), "\n")
    dropout_alphas <- slhs_net$fc1$get_dropout_rates()
    cat("alphas: ", round(as_array(dropout_alphas), 2), "\n \n")
  }


# }



contents <- list(
  "n_obs" = n_obs,
  "true_coefs" = true_coefs,
  "seed" = params$seed,
  "err_sig" = err_sig
)




















