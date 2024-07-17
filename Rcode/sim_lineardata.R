##################################################
## Project:   linear data sim
## Date:      Jul 17, 2024
## Author:    Arnie Seong
##################################################


library(here)
library(tidyr)
library(dplyr)

#competitors
library(BoomSpikeSlab)
library(glmnet)




library(torch)
source(here("Rcode", "BayesianLayers.R"))
source(here("Rcode", "sim_functions.R"))





n_obs <- 100
true_coefs = c(-0.5, 1, -2, 4, rep(0, times = 100))
train_epochs <- 10000

convergence_crit <- 1e-5
verbose <- TRUE
# loss moving average stopping criterion length
ma_length <- 50
loss_store_vec <- rep(NA, times = 2*ma_length)



lin_simdat <- sim_linear_data(
  n = n_obs,
  true_coefs = true_coefs
)

# # # # # # # # # # # # # # # # # # # # # # # # #
## BAYESIAN LAYER NJ ----
# # # # # # # # # # # # # # # # # # # # # # # # #

SLNJ <- nn_module(
  "SLNJ",
  initialize = function() {
    self$fc1 = BayesianLayerNJ(    
      in_features = lin_simdat$d_in, 
      out_features = 1,
      use_cuda = FALSE,
      init_weight = NULL,
      init_bias = NULL,
      clip_var = NULL
    )
  },
  
  forward = function(x) {
    x %>% 
      self$fc1() 
  },
  
  get_model_kld = function(){
    kl1 = self$fc1$get_kl()
    kld = kl1
    return(kld)
  }
)

slnj_net <- SLNJ()
slnj_net(lin_simdat$x)

optim_slnj <- optim_adam(slnj_net$parameters)
loss_diff <- 1
loss <- torch_zeros(1)
loss_store_mat <- array(NA, dim = c(3, 2*ma_length))
rownames(loss_store_mat) <- c("epoch", "loss", "loss_MA")
burn_in <- 100
loss_ma_stop <- FALSE

epoch <- 1
stop_criteria_reached <- FALSE

while (epoch < train_epochs & !stop_criteria_reached){
  prev_loss <- loss
  epoch <- epoch + 1
  
  y_pred <- slnj_net(lin_simdat$x)
  
  mse <- nnf_mse_loss(y_pred, lin_simdat$y)
  kl <- slnj_net$get_model_kld() / lin_simdat$n
  
  loss <- mse + kl
  loss_diff <- (loss - prev_loss)$item()
  
  if(epoch %% 1000 == 0 & verbose){
    cat(
      "Epoch: ", epoch, 
      "MSE + KL/n = ", mse$item(), " + ", kl$item(), 
      " = ", loss$item(), 
      "\n"
    )

  }
  
  if (epoch > burn_in) {
    # stopping criterion --- if loss MA hasn't gone down
    # in ma_length epochs, stop training
    store_ind <- epoch %% (2*ma_length) + 1
    loss_store_mat[1, store_ind] <- epoch
    loss_store_mat[2, store_ind] <- loss$item()

    # subset losses to contribute to MA
    if (store_ind == 2*ma_length){
      ma_inds <- c(2*ma_length, 1:(ma_length-1))
    } else if (store_ind > ma_length){
      ma_inds <- c(
        store_ind:(2 * ma_length),
        1:((store_ind-ma_length) %% ma_length)
      )[1:ma_length]
    } else {
      ma_inds <- store_ind:(store_ind + ma_length - 1)
    }

    losses_vec <- loss_store_mat[2, ma_inds]
    loss_store_mat[3, store_ind] <- mean(losses_vec, na.rm = TRUE)
    if (epoch > (burn_in + 2*ma_length)){
      loss_ma_vec <- loss_store_mat[3, ma_inds]
      loss_ma_diff <- diff(loss_ma_vec)
      losses_diff <- diff(losses_vec)
      if (sum(loss_ma_diff > 0 & losses_diff > 0) >= length(loss_ma_diff)) {
        loss_ma_stop <- TRUE
      }
    }

  }
  
  # zero out previous gradients
  optim_slnj$zero_grad()
  # backprop
  loss$backward()
  # update weights
  optim_slnj$step()
  
  stop_criteria_reached <- abs(loss_diff) < convergence_crit  |  loss_ma_stop
}

dropout_alphas <- slnj_net$fc1$get_log_dropout_rates()$exp()
coef_mse <- sum((slnj_net$fc1$compute_posterior_param()$post_weight_mu - true_coefs)^2)

slnj_keeps <- slnj_net$fc1$get_log_dropout_rates()$exp() < 0.05
slnj_bin_err <- binary_err_rate(est = as_array(slnj_keeps), tru = lin_simdat$true_coefs != 0 )

cat("coef_mse ", as_array(coef_mse), "\n")
cat("binary error: \n")
slnj_bin_err



# store: 
# time taken
# dopout_alphas
# coef_mse
# slnj_keeps
# slnj_bin_err
# epoch
# stop_criteria_reached




# # # # # # # # # # # # # # # # # # # # # # # # #
## lm ----
# # # # # # # # # # # # # # # # # # # # # # # # #

lin_simdat <- sim_linear_data(
  n = n_obs,
  true_coefs = true_coefs
)
get_lm_stats(simdat = lin_simdat)




# # # # # # # # # # # # # # # # # # # # # # # # #
## LASSO ----
# # # # # # # # # # # # # # # # # # # # # # # # #
x <- as_array(lin_simdat$x)
y <- as_array(lin_simdat$y)

cvfit <- glmnet::cv.glmnet(x, y)
lasso_coefs <- coef(cvfit, s = "lambda.1se")

binary_err_rate(
  est = lasso_coefs[-1]==0, 
  tru = true_coefs==0
)





# # # # # # # # # # # # # # # # # # # # # # # # #
## spike-slab ----
# # # # # # # # # # # # # # # # # # # # # # # # #
X <- cbind(1, x)
prior = IndependentSpikeSlabPrior(X, y, 
                                  expected.model.size = 1,
                                  prior.beta.sd = rep(1, ncol(X))) 
lm.ss = lm.spike(y ~ x, niter = 1000, prior = prior)
summary(lm.ss)$coef



# # # # # # # # # # # # # # # # # # # # # # # # #
## mombf ----
# # # # # # # # # # # # # # # # # # # # # # # # #



