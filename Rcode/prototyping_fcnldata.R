##################################################
## Project:   Prototyping doc
## Date:      Oct 23, 2024
## Author:    Arnie Seong
##################################################

# early conclusions - worth trying current model 
# 3-5 fully-connected NJBL
# only use KL from layer 1
# will also be interesting to see sparsity in other layers


library(here)
library(tidyr)
library(dplyr)
library(ggplot2)

# BayesCompress
library(torch)
source(here("Rcode", "BayesianLayers.R"))
source(here("Rcode", "sim_functions.R"))


# simulated data settings
n_obs <- 10000
sig <- 1
d_in <- 100
d_out <- 1
d_hidden1 <- 25
d_hidden2 <- 25
d_hidden3 <- 25
d_hidden4 <- 25
d_hidden5 <- 25
d_hidden6 <- 25



# # # # # # # # # # # # 
##    MLNJ training parameters

# set initial value for dropout rate alpha
# (default value is 1/2)
init_alpha <- 0.9
use_cuda <- cuda_is_available()
verbose <- TRUE
report_every <- 500
max_train_epochs <- 20000

# loss moving average stopping criterion length
ma_length <- 50
burn_in <- 300
convergence_crit <- 1e-6


test_train_split <- TRUE
test_train_ratio <- 0.2


# CV
use_cv <- TRUE
cv_k <- 5
refresh_cv_every <- 500


# # # # # # # # # # # # 
##    SIMULATION START    
# generate data
fcn1 <- function(x) exp(x/2) 
fcn2 <- function(x) 2*cos(pi*x) 
fcn3 <- function(x) abs(x)^(1.5) 
fcn4 <- function(x) cos(pi*x) + 2*sin(pi*x/1.5) - 2*x


# fcn_simdat <- sim_func_data(
#   n_obs = n_obs,
#   d_in = d_in,
#   flist = list(fcn1, fcn2, fcn3, fcn4), 
#   err_sigma = sig)


fcn_simdat <- sim_func_data_unifx(
  n_obs = n_obs,
  d_in = d_in,
  flist = list(fcn1, fcn2, fcn3, fcn4), 
  err_sigma = sig)



true_model <- c(
  rep(1, fcn_simdat$d_true), 
  rep(0, fcn_simdat$d_in - fcn_simdat$d_true)
)


## plotting functional relationships ----

x_grid <- -50:50 / 10
y1 <- fcn1(x_grid)
y2 <- fcn2(x_grid)
y3 <- fcn3(x_grid)
y4 <- fcn4(x_grid)

fcn_df <- data.frame(
  x_grid, y1, y2, y3, y4
)

names(fcn_df) <- c(
  "x",
  "f_1(x)",
  "f_2(x)",
  "f_3(x)",
  "f_4(x)"
)


fcn_df %>% 
  pivot_longer(cols = -x) %>% 
  ggplot() + 
  geom_line(
    aes(
      y = value,
      x = x,
      color = name
    )
  ) + 
  labs(
    title = "individual contributions of x1, x2, x3, x4 to response",
    color = ""
  )



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
      clip_var = 0.04
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
    #   out_features = d_hidden3,
    #   use_cuda = use_cuda,
    #   init_weight = NULL,
    #   init_bias = NULL,
    #   init_alpha = init_alpha,
    #   clip_var = NULL
    # )
    # 
    # self$fc4 = BayesianLayerNJ(
    #   in_features =  d_hidden3,
    #   out_features = d_hidden4,
    #   use_cuda = use_cuda,
    #   init_weight = NULL,
    #   init_bias = NULL,
    #   init_alpha = init_alpha,
    #   clip_var = NULL
    # )
    # 
    # self$fc5 = BayesianLayerNJ(
    #   in_features =  d_hidden4,
      out_features = d_out,
      use_cuda = use_cuda,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = init_alpha,
      clip_var = NULL
    )
    
    
    ## try using just regular dense layers after the input layer
    ## RESULTS:  PRETTY BAD.  MSE goes way down fast.  Probably just memorizing the data.
    # self$fc2 = nn_linear(
    #   in_features = d_hidden1,
    #   out_features = d_hidden2
    # )
    # 
    # self$fc3 = nn_linear(
    #   in_features = d_hidden2,
    #   out_features = d_hidden3
    # )
    # 
    # self$fc4 = nn_linear(
    #   in_features = d_hidden3,
    #   out_features = d_hidden4
    # )
    # 
    # self$fc5 = nn_linear(
    #   in_features = d_hidden4,
    #   out_features = d_out
    # )
    
    
    
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
    # kl4 = self$fc4$get_kl()
    # kl5 = self$fc5$get_kl()
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


## test-train split
if (test_train_split){
  end_train <- floor(n_obs * (1 - test_train_ratio))
  
  x_train <- fcn_simdat$x[1:end_train, ]
  y_train <- fcn_simdat$y[1:end_train, ]
  
  x_test <- fcn_simdat$x[(end_train + 1):n_obs, ]
  y_test <- fcn_simdat$y[(end_train + 1):n_obs, ]
  
  tt_mse <- data.frame(
    "train_mse" = rep(NA, times = max_train_epochs),
    "test_mse" = rep(NA, times = max_train_epochs),
    "epoch" = 1:max_train_epochs
  )
}



## Cross-val setup ----

gen_cv_inds <- function(n_obs, cv_k){
  rand_inds <- sample(1:n_obs, n_obs)
  fold_n <- floor(n_obs / cv_k)
  cv_inds <- list()
  for (i in 1:cv_k){
    if (i < cv_k){
      cv_inds[[i]] <- rand_inds[(i-1)*fold_n + 1:fold_n]
    } else if (i == cv_k){
      cv_inds[[cv_k]] <- rand_inds[((cv_k-1)*fold_n + 1):n_obs]
    }
  }
  return(cv_inds)
}

gen_cv_fold_inds <- function(cv_inds, fold_i){
  n_train <- length(do.call(c, cv_inds))
  test_inds <- cv_inds[[fold_i]]
  train_inds <- setdiff(1:n_train, test_inds)
  list(
    "train_inds" = train_inds,
    "test_inds" = test_inds
  )
}


epoch <- 0
if (test_train_split & use_cv){
  cv_inds <- gen_cv_inds(n_obs = end_train, cv_k)
} else if (test_train_split & !use_cv){
  x_train_i <- x_train
  y_train_i <- y_train
  x_test_i <- x_test
  y_test_i <- y_test  
}










# make arrays / df for plotting predicted functions
pred_mats <- make_pred_mats(
  flist = list(fcn1, fcn2, fcn3, fcn4), 
  xgrid = seq(-4.9, 5, length.out = 100), 
  d_in
)



## train loop ----
while (epoch < max_train_epochs & !converge_stop & !loss_ma_stop){
  prev_loss <- loss
  epoch <- epoch + 1

  if (test_train_split){
    
    if (use_cv){
      if (epoch %% refresh_cv_every == 1) {
        cv_inds <- gen_cv_inds(n_obs = end_train, cv_k)
      }
      
      fold_i <- epoch %% cv_k + 1
      fold_i_inds <- gen_cv_fold_inds(cv_inds, fold_i)
      
      x_train_i <- x_train[fold_i_inds$train_inds, ]
      y_train_i <- y_train[fold_i_inds$train_inds]
      
      x_test_i <- x_train[fold_i_inds$test_inds, ]
      y_test_i <- y_train[fold_i_inds$test_inds]
    }
    
    y_pred <- mlnj_net(x_train_i)
    mse <- nnf_mse_loss(y_pred, y_train_i)
    kl <- mlnj_net$get_model_kld() / end_train
    y_pred_test <- mlnj_net(x_test_i)
    mse_test <- nnf_mse_loss(y_pred_test, y_test_i)
    
    tt_mse[epoch, 1:2] <- c(mse$item(), mse_test$item())
    
  } else {
    y_pred <- mlnj_net(fcn_simdat$x)
    mse <- nnf_mse_loss(y_pred, fcn_simdat$y)
    kl <- mlnj_net$get_model_kld() / fcn_simdat$n_obs
  }
  
  
  loss <- mse + kl
  loss_diff <- (loss - prev_loss)$item()
  
  if (epoch %% report_every == 0 & verbose){
    
    txt <- paste0(
      "Epoch: ", epoch, 
      "\n MSE + KL/n = ", mse$item(), " + ", kl$item(), 
      " = ", loss$item(), 
      "\n"
    )
    cat_color(txt)
    
    log_alphas <- as_array(mlnj_net$fc1$get_log_dropout_rates())
    mlnj_keeps <- exp(log_alphas) < 0.05
    # print(round(log_alphas, 2)[1:15])
    print(round(exp(log_alphas)[1:15], 4))
    mlnj_bin_err <- binary_err(est = mlnj_keeps, tru = true_model)
    print(round(mlnj_bin_err, 4))
    cat("\n")
    
    if (test_train_split){
      print(tt_mse[epoch, ])
      
      tt_mse_plot <- na.omit(tt_mse) %>% 
        slice(which(row_number() %% floor(report_every/5) == 1))
      
      if (epoch > 1500) {
        tt_mse_plot <- tt_mse[1000:epoch,] %>% 
          slice(which(row_number() %% floor(report_every/5) == 1))
      }
      
      plt_preds <- plot_fcn_preds(
        torchmod = mlnj_net, 
        pred_mats = pred_mats) + 
        labs(subtitle=paste0(
          " d_in: ", d_in,
          "   |   n_obs: ", n_obs,
          "   |   epoch: ", epoch
          ))
      
      plt_mse <-  tt_mse_plot %>% 
        pivot_longer(cols = 1:2) %>% 
        ggplot() + 
          geom_line(
            aes(
              y = value,
              x = epoch,
              color = name
            )
          ) + 
        labs(
          title = "test/train MSE by epoch"
        )

      print(plt_mse)
      print(plt_preds)
      
    }

    
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












# stopping condition
tt_mse <- tt_mse %>% 
  mutate(diff_mse = test_mse - train_mse) %>% 
  mutate(diff_ma = zoo::rollmean(x = diff_mse, k = 50, fill = NA)) %>% 
  mutate(diff_mse_lag1 = diff_mse - lag(diff_mse)) %>% 
  mutate(test_ma = zoo::rollmean(x = test_mse, k = 50, fill = NA)) %>% 
  mutate(train_ma = zoo::rollmean(x = train_mse, k = 50, fill = NA))

  


tt_mse %>% 
  subset(epoch > 3000 & epoch < 8000) %>% 
  ggplot(aes(
    y = diff_mse_lag1,
    x = epoch
  )) + 
  geom_line() + 
  geom_smooth()


tt_mse %>% 
  subset(epoch > 3000 & epoch < 8000) %>% 
  ggplot(aes(
    y = diff_ma,
    x = epoch
  )) + 
  geom_line() + 
  geom_smooth()


tt_mse %>% 
  subset(epoch > 3000 & epoch < 8000) %>% 
  pivot_longer(cols=c("test_mse", "train_mse"), names_to="type") %>% 
  ggplot(aes(
    y = value,
    x = epoch,
    color = type
  )) + 
  geom_line() + 
  geom_smooth()


tt_mse %>% 
  subset(epoch > 3000 & epoch < 8000) %>% 
  pivot_longer(cols=c("test_ma", "train_ma"), names_to="type") %>% 
  ggplot(aes(
    y = value,
    x = epoch,
    color = type
  )) + 
  geom_line() 









