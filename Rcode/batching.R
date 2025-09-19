
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

fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
fcn3 <- function(x) abs(x)^(1.5)
fcn4 <- function(x) - (abs(x))
flist = list(fcn1, fcn2, fcn3, fcn4)
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
  "sim_name" = "hshoe, minibatched, fcnal data",
  "seed" = 12501,
  "n_sims" = 1, 
  "train_epochs" = 1e6, # 15E5,
  "report_every" = 1e4, # 1E4,
  "use_cuda" = use_cuda,
  "d_in" = 104,
  "d_hidden1" = 16,
  "d_hidden2" = 8,
  "d_out" = 1,
  "n_obs" = 12500,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "alpha_thresh" = 1 / qchisq(1 - (0.05 / 104), df = 1),
  "flist" = flist,
  "err_sig" = 1,
  "convergence_crit" = 1e-7,
  "ttsplit" = 4/5,
  "batch_size" = 2^5,
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



# verbose = TRUE
# want_plots = TRUE
# want_fcn_plots = TRUE
# save_fcn_plots = FALSE
# want_all_params = FALSE
# save_mod = TRUE
# save_mod_path_stem = NULL

# sim_hshoe ----
sim_hshoe <- function(
    sim_ind,    # to ease parallelization
    sim_params,     # same as before, but need to include flist
    nn_model,   # torch nn_module,
    verbose = TRUE,      # provide updates in console
    want_plots = TRUE,   # provide graphical updates of KL, MSE
    want_fcn_plots = TRUE, # display predicted functions
    save_fcn_plots = FALSE,
    want_all_params = FALSE,
    save_mod = TRUE,
    save_mod_path_stem = NULL
){
  ## generate data ----
  set.seed(sim_params$sim_seeds[sim_ind])
  torch_manual_seed(sim_params$sim_seeds[sim_ind])
  
  simdat <- sim_func_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    err_sigma = sim_params$err_sig
  )
  if (sim_params$use_cuda){
    simdat$x <- simdat$x$to(device = "cuda")
    simdat$y <- simdat$y$to(device = "cuda")
  }
  
  ## initialize BNN & optimizer ----
  model_fit <- nn_model()
  optim_model_fit <- optim_adam(model_fit$parameters)
  
  if (save_mod){
    if(is.null(save_mod_path_stem)){
      save_mod_path_stem <- here::here("sims", 
                                       "results", 
                                       paste0("fcnl_hshoe_mod_", 
                                              sim_params$n_obs, "obs_", 
                                              sim_params$sim_seeds[sim_ind]
                                       ))
    }
    save_mod_path <- paste0(save_mod_path_stem, ".pt")
  }
  
  
  
  ## set up plotting while training: ----
  # original function plots
  xshow <- seq(-3, 3, length.out = 100)
  yshow <- sapply(sim_params$flist, function(fcn) fcn(xshow))
  colnames(yshow) <- paste0("f", 1:ncol(yshow))
  
  orig_func_df <- data.frame(
    yshow,
    "x"  = xshow
  ) %>% 
    pivot_longer(cols = -x, names_to = "fcn")
  
  # to feed into BNN model during training loop updates
  curvmat <- matrix(0, ncol = length(sim_params$flist), nrow = length(sim_params$flist) * 100)
  for (i in 1:length(flist)){
    curvmat[1:length(xshow) + (i-1) * length(xshow), i] <- xshow
  }
  mat0 <- matrix(0, nrow = nrow(curvmat), ncol = sim_params$d_in - length(sim_params$flist))
  x_plot <- torch_tensor(cbind(curvmat, mat0), device = ifelse(sim_params$use_cuda, "cuda", "cpu"))
  y_plot <- model_fit(x_plot)  # need to add deterministic argument
  plotmat <- cbind(scale(as_array(y_plot)), curvmat)
  colnames(plotmat) <- c("y", paste0("f", 1:length(sim_params$flist)))
  plotdf <- as.data.frame(plotmat)
  
  
  # store: # train, test mse and kl ----
  report_epochs <- seq(
    sim_params$report_every, 
    sim_params$train_epochs, 
    by = sim_params$report_every
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
    ncol = sim_params$d_in
  )
  rownames(alpha_mat) <- report_epochs
  
  # store: weight posterior params
  if (want_all_params){
    w_mu_mat <-
      w_var_mat <- 
      marginal_dropout_mat <-
      atilde_mu_mat <- 
      btilde_mu_mat <-
      atilde_logvar_mat <-
      btilde_logvar_mat <- alpha_mat
    
    global_dropout_vec <-   
      tau_vec <- 
      sa_mu_vec <-
      sb_mu_vec <- 
      sa_logvar_vec <- 
      sb_logvar_vec <- rep(NA, length(report_epochs))
  }
  
  
  ## SETUP test-train split and minibatching----
  ttsplit_ind <- floor(sim_params$n_obs * sim_params$ttsplit)
  x_train <- simdat$x[1:ttsplit_ind, ] 
  y_train <- simdat$y[1:ttsplit_ind, ]
  x_test <- simdat$x[(ttsplit_ind+1):sim_params$n_obs, ] 
  y_test <- simdat$y[(ttsplit_ind+1):sim_params$n_obs, ]

  num_batches <- ttsplit_ind %/% sim_params$batch_size
  batch_inds_vec <- 1:ttsplit_ind
  
  ## TRAINING LOOP ----
  ## initialize training params
  epoch <- 1
  loss <- torch_tensor(1, device = dev_select(sim_params$use_cuda))
  
  ## stop criteria
  stop_criteria_met <- FALSE
  stop_epochs <- c()
  ## test_mse_storage
  mse_test <- torch_tensor(0, device = dev_select(sim_params$use_cuda))
  loss_test <- torch_tensor(1, device = dev_select(sim_params$use_cuda)) 
  test_mse_store <- rep(0, times = sim_params$stop_k + 1)
  test_mse_streak <- rep(FALSE, times = sim_params$stop_streak)
  
  
  while (!stop_criteria_met){
    batch_num <- epoch %% num_batches
    # reshuffle batches if batch_num == 0
    if (batch_num == 0) {batch_inds_vec <- sample(1:ttsplit_ind)}
    batch_inds <- batch_inds_vec[1:sim_params$batch_size + (batch_num)*sim_params$batch_size]

    # fit & metrics
    yhat_train <- model_fit(x_train[batch_inds, ])
    mse <- nnf_mse_loss(yhat_train, y_train[batch_inds])
    kl <- model_fit$get_model_kld() / sim_params$batch_size
    loss <- mse + kl
    
    
    # gradient step 
    # zero out previous gradients
    optim_model_fit$zero_grad()
    # backprop
    loss$backward()
    # update weights
    optim_model_fit$step()


    # if (epoch > (sim_params$burn_in - 2 * (sim_params$stop_k + sim_params$stop_streak))) {
    #   test_mse_store <- roll_vec(test_mse_store, as_array(mse_test))
    # }
# 
#     # check test_mse stop criteria:
#     # abs(diff(mse)) < sd(mse) for streak epochs in a row,
#     # sd calculated using last stop_k epochs
#     if (epoch > sim_params$burn_in){
#       test_mse_sd <- sd(test_mse_store[1:sim_params$stop_k])
#       test_mse_absdiff <- abs(diff(test_mse_store[sim_params$stop_k + 0:1]))
#       test_mse_compare <- test_mse_absdiff < test_mse_sd
#       test_mse_streak <- roll_vec(test_mse_streak, test_mse_compare)
#       test_mse_stopcrit <- all(test_mse_streak)
#       
#       if (test_mse_stopcrit){
#         stop_epochs <- c(stop_epochs, epoch)
#       }
#     }
    
    # store results (every `report_every` epochs) ----
    time_to_report <- epoch!=0 & (epoch %% sim_params$report_every == 0)
    if (time_to_report){
      row_ind <- epoch %/% sim_params$report_every
      
      # compute test loss 
      yhat_test <- model_fit(x_test)
      mse_test <- nnf_mse_loss(yhat_test, y_test)
      
      loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_test$item())
      dropout_alphas <- model_fit$fc1$get_dropout_rates()
      alpha_mat[row_ind, ] <- as_array(dropout_alphas)
      
      # storing other optional parameters, mostly for diagnostics
      if (want_all_params){
        global_dropout_vec[row_ind] <- as_array(model_fit$fc1$get_dropout_rates(type = "global"))
        marginal_dropout_mat[row_ind, ] <- as_array(model_fit$fc1$get_dropout_rates(type = "marginal"))
        tau_vec[row_ind] <- as_array(model_fit$fc1$tau)
        w_mu_mat[row_ind, ] <- as_array(model_fit$fc1$compute_posterior_param()$post_weight_mu)
        w_var_mat[row_ind, ] <- as_array(model_fit$fc1$compute_posterior_param()$post_weight_var)
        
        atilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_mu)
        btilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_mu)
        atilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_logvar)
        btilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_logvar)
        sa_mu_vec[row_ind] <- as_array(model_fit$fc1$sa_mu)
        sa_logvar_vec[row_ind] <- as_array(model_fit$fc1$sa_logvar)
        sb_mu_vec[row_ind] <- as_array(model_fit$fc1$sb_mu)
        sb_logvar_vec[row_ind] <- as_array(model_fit$fc1$sb_logvar)
      }
    } # end result storing and updating
    
    
    # in-console and graphical training updates ----
    if (time_to_report & verbose){
      cat(
        "Epoch:", epoch,
        "MSE + KL/n =", mse$item(), "+", kl$item(),
        "=", loss$item(),
        "\n",
        "train mse:", round(mse$item(), 4), 
        "; test_mse:", round(mse_test$item(), 4),
        "\n", 
        sep = " "
      )
      cat("alphas: ", round(as_array(dropout_alphas), 2), "\n")
      display_alphas <- ifelse(
        as_array(dropout_alphas) <= sim_params$alpha_thresh,
        round(as_array(dropout_alphas), 3),
        "."
      )
      cat("alphas below ", round(sim_params$alpha_thresh, 4), ": ")
      cat(display_alphas, sep = " ")
      
      # if (length(stop_epochs > 0)){
      #   stop_msg <- paste0(
      #     "\n ********************************************* \n",
      #     "test_mse STOP CONDITION reached ", 
      #     length(stop_epochs), " times; ", 
      #     "min / max: ", min(stop_epochs), " / ", max(stop_epochs),
      #     "\n ********************************************* \n"
      #   )
      #   cat_color(stop_msg)
      # }
      cat(" \n \n")
      
      
      # graphical training updates
      if (want_plots & row_ind > 5){
        
        # only show most recent (for scale of plot)
        start_plot_row_ind <- 1
        if (row_ind >= 20){
          start_plot_row_ind <- row_ind - 10
        } else if (row_ind >= 60){
          start_plot_row_ind <- row_ind - 30
        } else if (row_ind >= 100){
          start_plot_row_ind <- row_ind - 50
        }
        # set up plotting df
        temp_lmat <- cbind(
          loss_mat[start_plot_row_ind:row_ind,],
          "epoch" = report_epochs[start_plot_row_ind:row_ind]
        )
        
        mse_plotdf <- temp_lmat %>% 
          data.frame() %>% 
          pivot_longer(
            cols = -epoch,
            names_to = "metric"
          )
        
        # plot MSE vs epochs
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
        
        # plot KL vs epochs
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
        grid_plt <- gridExtra::grid.arrange(
          grobs = list(mse_plt, kl_plt),
          nrow = 2
        )
        print(grid_plt)
      } # end graphical training updates (want_plots = TRUE)
    } # end training updates (verbose = TRUE)
    
    # function plots
    if (time_to_report & want_fcn_plots){
      plotdf$y <- scale(as_array(model_fit(x_plot)))
      
      plt <- plotdf %>% 
        pivot_longer(cols = -y, values_to = "x", names_to = "fcn") %>%
        ggplot(aes(y = y, x = x, color = fcn)) + 
        geom_line() + 
        geom_line(
          data = orig_func_df,
          linetype = "dotted",
          aes(y = value, x = x, color = fcn)
        ) + 
        labs(
          title = "predicted (solid) and original (dotted) functions",
          subtitle = paste0("epoch: ", epoch)
        )
      
      if (save_fcn_plots){
        save_plt_path <- paste0(save_mod_path_stem, "_e", epoch, ".png")
        ggsave(filename = save_plt_path, plot = plt, height = 4, width = 6)
      } else {
        print(plt)
      }
    } # end function update plots (want_fcn_plots = TRUE)
    
    
    # increment epoch counter
    epoch <- epoch + 1
    stop_criteria_met <- epoch > sim_params$train_epochs
  } # end training WHILE loop
  
  # compile results ----
  sim_res <- list(
    "sim_ind" = sim_ind,
    # "stop_epochs" = stop_epochs,
    "fcn_plt" = plt,
    "loss_mat" = loss_mat,
    "alpha_mat" = alpha_mat
  )
  
  if (want_all_params){
    # add optionally stored network params
    sim_res$w_mu_mat <- w_mu_mat
    sim_res$w_var_mat <- w_var_mat
    sim_res$global_dropout_vec <- global_dropout_vec
    sim_res$marginal_dropout_mat <- marginal_dropout_mat
    sim_res$atilde_mu_mat <- atilde_mu_mat
    sim_res$btilde_mu_mat <- btilde_mu_mat
    sim_res$atilde_logvar_mat <- atilde_logvar_mat
    sim_res$btilde_logvar_mat <- btilde_logvar_mat
    sim_res$sa_mu_vec <- sa_mu_vec
    sim_res$sb_mu_vec <- sb_mu_vec
    sim_res$sa_logvar_vec <- sa_logvar_vec
    sim_res$sb_logvar_vec <- sb_logvar_vec
    sim_res$tau_vec <- tau_vec
  } 
  
  # notify completed training ----
  completed_msg <- paste0(
    "\n \n ******************** \n ******************** \n",
    "sim #", 
    sim_ind, 
    " completed \n ",
    "final alphas below threshold: ",
    paste0(display_alphas, collapse = " "),
    "\n ******************** \n ******************** \n \n"
  )
  cat_color(txt = completed_msg)
  
  # save torch model ----
  if (save_mod){
    torch_save(model_fit, path = save_mod_path)
    cat_color(txt = paste0("model saved: ", save_mod_path))
    sim_res$mod_path = save_mod_path
  }
  return(sim_res)
}


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




