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
  "n_obs" = 100,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "seed" = 314,
  "err_sig" = 1,
  "verbose" = TRUE,
  "report_every" = 1000, # training epochs between display/store results
  "want_plots" = TRUE,   # set to FALSE when running in parallel
  "train_epochs" = 200000,
  "convergence_crit" = 1e-7,
  "ttsplit" = 4/5,
  "stop_criteria" = "convergence" 
  # OPTIONS:  ttloss: train/test loss widening, 
  #           convergence: loss diff between epochs,
  #           MA loss increasing over 50(?) epochs
)


## define model
SLHS <- nn_module(
  "SLHS",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = d_in, 
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
  nrow = length(length(report_epochs)),
  ncol = params$d_in
)
rownames(alpha_mat) <- report_epochs




## initialize training params ----
loss_diff <- 1
loss <- torch_zeros(1)
epoch <- 1
stop_criteria <- FALSE


ttsplit_ind <- floor(params$n_obs * params$ttsplit)
train_x <- lin_simdat$x[1:ttsplit_ind, ]
test_x <- lin_simdat$x[(ttsplit_ind+1):params$n_obs, ]
slhs_net(train_x)
optim_slhs <- optim_adam(slhs_net$parameters)



while (epoch < params$train_epochs 
       & abs(loss_diff) > params$convergence_crit){
  prev_loss <- loss
  epoch <- epoch + 1

  y_pred <- slhs_net(lin_simdat$x)

  mse <- nnf_mse_loss(y_pred, lin_simdat$y)
  kl <- slhs_net$get_model_kld() / lin_simdat$n

  loss <- mse + kl
  loss_diff <- (loss - prev_loss)$item()


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

  # zero out previous gradients
  optim_slhs$zero_grad()
  # backprop
  loss$backward()
  # update weights
  optim_slhs$step()
}



contents <- list(
  "n_obs" = n_obs,
  "true_coefs" = true_coefs,
  "seed" = params$seed,
  "err_sig" = err_sig
)




















