##################################################
## Project:   MLP functional data sim
## Date:      Aug 14, 2024
## Author:    Arnie Seong
##################################################

library(here)
library(tidyr)
library(dplyr)

# BayesCompress
library(torch)
source(here("Rcode", "BayesianLayers.R"))
source(here("Rcode", "sim_functions.R"))

#competitors
library(BoomSpikeSlab)
library(glmnet)




# sim params --------------------------------------------------------------
testing <- TRUE

fname <- ifelse(testing, "MLP_fcnl_sim_sig1_TEST.Rdata", "MLP_fcnl_sim_sig1.Rdata")
fpath <- here("Rcode", "results", fname)

n_sims <- ifelse(testing, 3, 100)

res <- sapply(1:n_sims, function(x) NULL)

# simulated data settings
n_obs <- 10000
sig <- 1
d_in <-  100
d_hidden1 <- 50
d_hidden2 <- 25





# # # # # # # # # # # # 
##    MLNJ training parameters

# set initial value for dropout rate alpha
# (default value is 1/2)
init_alpha <- 0.5
use_cuda <- cuda_is_available()
max_train_epochs <- ifelse(testing, 500, 40000)
verbose <- testing
burn_in <- ifelse(testing, 100, 10000)
convergence_crit <- 1e-6
# loss moving average stopping criterion length
ma_length <- 50




for(sim_num in 1:n_sims){
  
  # # # # # # # # # # # # 
  ##    SIMULATION START    
  # generate data
  fcn1 <- function(x) exp(x/2)
  fcn2 <- function(x) cos(pi*x)
  fcn3 <- function(x) abs(x)^(1.5)
  fcn4 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x
  
  fcn_simdat <- sim_func_data(
    n_obs = n_obs,
    d_in = 100,
    flist = list(fcn1, fcn2, fcn3, fcn4), 
    err_sigma = 1)

  true_model <- c(
    rep(1, fcn_simdat$d_true), 
    rep(0, fcn_simdat$d_in-fcn_simdat$d_true)
  )
  
  # plot functions
  # xsamp <- -500:500 / 100
  # library(ggplot2)
  # qplot(y = fcn1(xsamp), x = xsamp)
  # qplot(y = fcn2(xsamp), x = xsamp)
  # qplot(y = fcn3(xsamp), x = xsamp)
  # qplot(y = fcn4(xsamp), x = xsamp)

  
  # # # # # # # # # # # # # # # # # # # # # # # # #
  ## BAYESIAN LAYER NJ ----
  # # # # # # # # # # # # # # # # # # # # # # # # #
  
  MLNJ <- nn_module(
    "MLNJ",
    initialize = function() {
      self$fc1 = BayesianLayerNJ(    
        in_features = fcn_simdat$d_in, 
        out_features = d_hidden1,
        use_cuda = use_cuda,
        init_weight = NULL,
        init_bias = NULL,
        init_alpha = init_alpha,
        clip_var = NULL
      )
      
      self$fc2 = BayesianLayerNJ(    
        in_features = d_hidden1, 
        out_features = d_hidden2,
        use_cuda = use_cuda,
        init_weight = NULL,
        init_bias = NULL,
        init_alpha = init_alpha,
        clip_var = NULL
      )
      
      self$fc3 = BayesianLayerNJ(    
        in_features =  d_hidden2, 
        out_features = 1,
        use_cuda = use_cuda,
        init_weight = NULL,
        init_bias = NULL,
        init_alpha = init_alpha,
        clip_var = NULL
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
  
  mlnj_net <- MLNJ()
  optim_mlnj <- optim_adam(mlnj_net$parameters)
  
  
  # initialize stopping criteria params
  # 3 stopping criteria:
  #    1: epochs > max_train_epochs
  #    2: loss_diff < convergence crit
  #    3: loss moving average (average over last ma_length epochs) 
  #       has increased for the last ma_length epochs.
  #       In this case, best model occurred ma_length epochs ago.
  loss_diff <- 1
  loss <- torch_zeros(1)
  loss_ma_stop <- FALSE
  converge_stop <- FALSE
  
  # track loss, alphas for stopping criteria
  loss_store_mat <- array(NA, dim = c(3, 2*ma_length))
  rownames(loss_store_mat) <- c("epoch", "loss", "loss_MA")
  log_alpha_mat <- array(NA, dim = c(2*ma_length, d_in + 4))
  colnames(log_alpha_mat) <- c(
    paste0("x", 1:d_in),
    "epoch",
    "mse",
    "kl",
    "coef_mse"
  )
  
  epoch <- 0
  while (epoch < max_train_epochs & !converge_stop & !loss_ma_stop){
    prev_loss <- loss
    epoch <- epoch + 1
    
    y_pred <- mlnj_net(fcn_simdat$x)
    
    mse <- nnf_mse_loss(y_pred, fcn_simdat$y)
    kl <- mlnj_net$get_model_kld() / fcn_simdat$n_obs
    
    loss <- mse + kl
    loss_diff <- (loss - prev_loss)$item()
    
    if(epoch %% 1000 == 0 & verbose){
      cat(
        "Epoch: ", epoch, 
        "MSE + KL/n = ", mse$item(), " + ", kl$item(), 
        " = ", loss$item(), 
        "\n"
      )
      mlnj_keeps <- mlnj_net$fc1$get_log_dropout_rates()$exp() < 0.05
      mlnj_bin_err <- binary_err(est = as_array(mlnj_keeps), tru = true_model)
      cat("binary error: \n")
      cat(round(mlnj_bin_err, 4))
      cat("\n")
      
    }
    
    if (epoch > burn_in) {
      # stopping criterion --- if loss MA hasn't decreased
      # in ma_length epochs, stop training
      store_ind <- epoch %% (2*ma_length) + 1
      loss_store_mat[1, store_ind] <- epoch
      loss_store_mat[2, store_ind] <- loss$item()
      log_alpha_mat[store_ind, ] <- c(
        as_array(mlnj_net$fc1$get_log_dropout_rates()),
        epoch,
        mse$item(), 
        kl$item(),
        NA
      )
      
      # store loss moving average
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
        
        converge_stop <- abs(loss_diff) < convergence_crit
        
        loss_ma_vec <- loss_store_mat[3, ma_inds]
        loss_ma_diff <- diff(loss_ma_vec)
        if (sum(loss_ma_diff > 0) >= length(loss_ma_diff)) {
          loss_ma_stop <- TRUE
        }
      }
      
    }
    
    
    # zero out previous gradients
    optim_mlnj$zero_grad()
    # backprop
    loss$backward()
    # update weights
    optim_mlnj$step()
    
  }
  
  # best model's alphas / other metrics
  if (loss_ma_stop) {
    best_epoch_row <- which(log_alpha_mat[, d_in + 1] == epoch - (ma_length))
  } else {
    best_epoch_row <- which(log_alpha_mat[, d_in + 1] == epoch)
  }
  
  # in case which returns "integer(0)" above:
  if (length(best_epoch_row) == 0){
    best_epoch_row <- store_ind
  }
  
  log_dropout_alphas <- log_alpha_mat[best_epoch_row, 1:d_in]
  other_metrics <- c(log_alpha_mat[best_epoch_row, d_in + 1:4])
  stop_reason <- c(
    "max_epochs" = epoch > max_train_epochs,
    "converge_crit" = converge_stop,
    "loss_ma" = loss_ma_stop
  ) 
  
  
  
  if (verbose){
    mlnj_keeps <- exp(log_dropout_alphas) < 0.05
    mlnj_bin_err <- binary_err(est = mlnj_keeps, tru = true_model)
    
    cat("binary error: \n")
    mlnj_bin_err
  }
  
  
  
  # store results: 
  # epoch
  # stop_reason
  # dropout_alphas
  # mse
  mlnj_res <- list(
    "epoch" = epoch,
    "stop_reason" = stop_reason,
    "log_dropout_alphas" = log_dropout_alphas,
    "log_alpha_mat" = log_alpha_mat,
    "other_metrics" = other_metrics
  )
  
  res[[sim_num]]$mlnj <- mlnj_res
  
  
  
  # # # # # # # # # # # # # # # # # # # # # # # # #
  ## lm ----
  # # # # # # # # # # # # # # # # # # # # # # # # #
  lm_res <- get_lm_stats(simdat = fcn_simdat)
  res[[sim_num]]$lm <- lm_res
  
  
  
  # # # # # # # # # # # # # # # # # # # # # # # # #
  ## LASSO ----
  # # # # # # # # # # # # # # # # # # # # # # # # #
  x <- as_array(fcn_simdat$x)
  y <- as_array(fcn_simdat$y)
  
  cvfit <- glmnet::cv.glmnet(x, y)
  lasso_coefs <- coef(cvfit, s = "lambda.1se")
  lasso_mse <- cvfit$cvm[cvfit$lambda == cvfit$lambda.1se]
  lasso_bin_err <- binary_err(est = !(lasso_coefs[-1]==0), tru = true_model)
  
  res[[sim_num]]$lasso <- list(
    "coefs" = lasso_coefs,
    "binary_err" = lasso_bin_err,
    "fit_mse" = lasso_mse,
    "coef_mse" = NA
  )
  
  
  
  
  
  # # # # # # # # # # # # # # # # # # # # # # # # #
  ## spike-slab ----
  # # # # # # # # # # # # # # # # # # # # # # # # #
  X <- cbind(1, x)
  prior = IndependentSpikeSlabPrior(X, y, 
                                    expected.model.size = 1,
                                    prior.beta.sd = rep(1, ncol(X))) 
  lm_ss = lm.spike(y ~ x, niter = 1000, prior = prior)
  ss_allcoefs_unordered <- summary(lm_ss)$coef
  
  # reorder results
  ss_intercept <- ss_allcoefs_unordered[rownames(ss_allcoefs_unordered)=="(Intercept)",]
  ss_coefs_unordered <- ss_allcoefs_unordered[rownames(ss_allcoefs_unordered)!="(Intercept)",]
  coef_order <- as.numeric(sapply(strsplit(rownames(ss_coefs_unordered), "x"), function(X) X[2]))
  ss_allcoefs <- rbind(
    "(Intercept)" = ss_intercept,
    ss_coefs_unordered[order(coef_order),]
  )
  ss_coefs <- ss_allcoefs[-1, ]
  ss_bin_err <- binary_err(
    est = ss_coefs[, 4] > 0.05, 
    tru = true_model)
  
  # get median mse from last quarter of draws
  ss_mse <- median(lm_ss$sse[751:1000]) / n_obs
  
  
  ss_res <- list(
    "coefs" = ss_coefs,
    "binary_err" = ss_bin_err,
    "fit_mse" = ss_mse,
    "coef_mse" = NA
  )
  
  
  res[[sim_num]]$ss <- ss_res
  
  # # # # # # # # # # # # # # # # # # # # # # # # #
  ## mombf ----
  # # # # # # # # # # # # # # # # # # # # # # # # #
  
  
  
  
  
  # # # # # # # # # # # # 
  ##    partial save
  
  if (sim_num %% 25 == 0){
    save(res, true_model, file = fpath)
    txt <- paste0("finished ", sim_num, " of ", n_sims)
    cat_color(txt)
  }
}

save(res, true_model, file = fpath)

