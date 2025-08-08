
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

## params
#    check whenever changing setting (testing / single vs parallel, etc) ##
#           n_sims, verbose, want_plots, train_epochs
params <- list(
  "sim_name" = "horseshoe, linear regression setting, unscaled KL",
  "n_sims" = 5, 
  "d_in" = 104,
  "n_obs" = 125,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "seed" = 314,
  "err_sig" = 1,
  "verbose" = TRUE,      # set to FALSE when running in parallel
  "report_every" = 1000, # training epochs between display/store results
  "want_plots" = TRUE,   # set to FALSE when running in parallel
  "train_epochs" = 200000,
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
set.seed(params$seed)
params$sim_seeds <- floor(runif(n = params$n_sims, 0, 1000000))


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



#### DISTRIBUTED FROM HERE ----
sim_ind <- 1

## generate linear data
set.seed(params$sim_seeds[sim_ind])
torch_manual_seed(params$sim_seeds[sim_ind])

lin_simdat <- sim_linear_data(
  n = params$n_obs,
  true_coefs = params$true_coefs,
  err_sigma = params$err_sig
)

# store: # train, test mse and kl
report_epochs <- seq(
  params$report_every, 
  params$train_epochs, 
  by = params$report_every
)

loss_mat <- matrix(
  NA, 
  nrow = length(report_epochs),
  ncol = 3
)
colnames(loss_mat) <- c("kl", "mse_train", "mse_test")
rownames(loss_mat) <- report_epochs


# store: alphas
alpha_mat <- matrix(
  NA, 
  nrow = length(report_epochs),
  ncol = params$d_in
)
rownames(alpha_mat) <- report_epochs

# store: weight posterior params
w_mu_mat <- w_var_mat <- alpha_mat

# store: alpha_tilde, beta_tilde?   s_a, s_b, tau? ---- 



## SETUP STOP CRITERIA ----
#### 
####  NOT FULLY IMPLEMENTED IN THIS FILE.  ONLY FOR STORAGE / VIEWING. 
####  ONLY STOPPING CRITERIA HERE IS MAX_EPOCHS.
####
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
  loss_test <- torch_zeros(1)  # set initial value
  loss_diff_test <- 1          # set initial value    
}

x_train <- lin_simdat$x[1:ttsplit_ind, ]
y_train <- lin_simdat$y[1:ttsplit_ind, ]

## initialize BNN & optimizer ----
slhs_net <- SLHS()
optim_slhs <- optim_adam(slhs_net$parameters)

## TRAINING LOOP ----
## initialize training params
loss_diff <- 1
loss <- torch_zeros(1)
epoch <- 0
stop_criteria_met <- FALSE

while (!stop_criteria_met){
  prev_loss <- loss
  epoch <- epoch + 1
  stop_criteria_met <- epoch > params$train_epochs
  
  
  # fit & metrics
  yhat_train <- slhs_net(x_train)
  mse <- nnf_mse_loss(yhat_train, y_train)
  kl <- slhs_net$get_model_kld() / log(ttsplit_ind)
  loss <- mse + kl
  # loss_diff <- loss - prev_loss
  
  
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
  }
  
  
  # store results (every `params$report_every` epochs)
  if (epoch %% params$report_every == 0 & epoch!=0){
    row_ind <- epoch %/% params$report_every
    
    loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_test$item())
  
    dropout_alphas <- slhs_net$fc1$get_dropout_rates()
    w_mu <- slhs_net$fc1$compute_posterior_param()$post_weight_mu
    w_var <- slhs_net$fc1$compute_posterior_param()$post_weight_var
    
    alpha_mat[row_ind, ] <- as_array(dropout_alphas)
    w_mu_mat[row_ind, ] <- as_array(w_mu)
    w_var_mat[row_ind, ] <- as_array(w_var)
    
    # in-console updates
    if (params$verbose){
      cat(
        "Epoch:", epoch,
        "MSE + KL/n =", mse$item(), "+", kl$item(),
        "=", loss$item(),
        "\n",
        "train mse:", round(mse$item(), 4), 
        "; test_mse:", round(mse_test$item(), 4),
        sep = " "
      )
      cat("alphas: ", round(as_array(dropout_alphas), 2), "\n \n")
    }
    
    if (params$want_plots & row_ind > 5){
      
      # don't want to always show the "burn-in" period
      start_plot_row_ind <- 1
      if (row_ind > 20){
        start_plot_row_ind <- 10
      } else if (row_inds > 60){
        start_plot_row_ind <- row_inds - 50
      }
      
      # set up plotting df
      temp_lmat <- cbind(
        loss_mat[start_plot_row_ind:row_ind,],
        "epoch" = report_epochs[start_plot_row_ind:row_ind]
      )
      # mserange <- range(temp_lmat[, 2:3])
      # klrange <- range(temp_lmat[,1])
      # scale_factor <- diff(mserange)/diff(klrange)
      # scaled_kl <- (temp_lmat[,1] - min(klrange)) * scale_factor + min(mserange)
      # temp_lmat[, 1] <- scaled_kl
      
      mse_plotdf <- temp_lmat %>% 
        data.frame() %>% 
        pivot_longer(
          cols = -epoch,
          names_to = "metric"
        )
      
      # MSE train vs test, and KL
      
      mse_plt <- mse_plotdf %>% 
        filter(metric != "kl") %>% 
        ggplot(aes(y = value, x = epoch, color = metric)) + 
        geom_line() + 
        labs(
          subtitle = paste0("MSE for epochs ", 
                         report_epochs[start_plot_row_ind],
                         " through ",
                         report_epochs[row_ind])
        )

      # KL
      kl_plt <- mse_plotdf %>%
        filter(metric == "kl") %>%
        ggplot(aes(y = value, x = epoch, color = metric)) +
        geom_line() +
        labs(
          subtitle = paste0("KL for epochs ",
                         report_epochs[start_plot_row_ind],
                         " through ",
                         report_epochs[row_ind])
        )
      gridExtra::grid.arrange(
        grobs = list(mse_plt, kl_plt),
        nrow = 2
      )
    }
  }
  
}



contents <- list(
  params
)










