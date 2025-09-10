sim_fcn_hshoe_fcnaldata <- function(
    sim_ind,    # to ease parallelization
    sim_params,     # same as before, but need to include flist
    nn_model,   # torch nn_module,
    train_epochs = 10000,
    verbose = TRUE,      # provide updates in console
    display_alpha_thresh = 0.1,    # display alphas below this number
    report_every = 1000, # training epochs between display/store results
    want_plots = TRUE,   # provide graphical updates of KL, MSE
    want_fcn_plots = TRUE, # display predicted functions
    want_all_params = FALSE,
    save_mod = TRUE,
    save_mod_path = NULL,
    stop_k = 100,
    stop_streak = 25,
    burn_in = 5E4
){
  # - fcn used with lapply for parallel processing
  # - basic linear regression setting
  # - only stop criteria used is max epochs
  # - if want_all_params = FALSE, only returns
  #     loss_mat (KL, mse vs epoch) and
  #     alpha_mat (local dropout rates vs epoch)
  # USAGE EXAMPLE:
  # > sim_params <- list(
  # >   "sim_name" = "horseshoe, linear regression setting, KL scaled by n",
  # >   "n_sims" = 100,
  # >   "d_in" = 104,
  # >   "n_obs" = 125,
  # >   "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  # >   "seed" = 314,
  # >   "err_sig" = 1,
  # >   "ttsplit" = 4/5,
  # >   "stop_criteria" = c("test_train","ma_loss_increasing")
  # > )
  # >
  # > res <- lapply(
  # >   1:sim_params$n_sims, 
  # >   function(X) sim_fcn_hshoe_linreg(
  # >     sim_ind = X, 
  # >     sim_params = sim_params,
  # >     nn_model = SLHS,  #### this needs to be pre-defined via torch::nn_module()
  # >     train_epochs = 1000,
  # >     verbose = TRUE,
  # >     report_every = 100,
  # >     want_plots = TRUE,
  # >     want_all_params = FALSE,
  # >     want_data = FALSE
  # >   )
  # > )
  
  ## generate data ----
  set.seed(sim_params$sim_seeds[sim_ind])
  torch_manual_seed(sim_params$sim_seeds[sim_ind])
  
  simdat <- sim_func_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    err_sigma = sim_params$err_sig
  )
  
  
  ## initialize BNN & optimizer ----
  model_fit <- nn_model()
  optim_slhs <- optim_adam(model_fit$parameters)
  
  
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
  
  # to feed into BNN model
  curvmat <- matrix(0, ncol = length(sim_params$flist), nrow = length(sim_params$flist) * 100)
  for (i in 1:length(flist)){
    curvmat[1:length(xshow) + (i-1) * length(xshow), i] <- xshow
  }
  mat0 <- matrix(0, nrow = nrow(curvmat), ncol = sim_params$d_in - length(sim_params$flist))
  x_plot <- torch_tensor(cbind(curvmat, mat0))
  y_plot <- model_fit(x_plot)  # need to add deterministic argument
  plotmat <- cbind(scale(as_array(y_plot)), curvmat)
  colnames(plotmat) <- c("y", paste0("f", 1:length(sim_params$flist)))
  plotdf <- as.data.frame(plotmat)
  
  
  # store: # train, test mse and kl ----
  report_epochs <- seq(
    report_every, 
    train_epochs, 
    by = report_every
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
  
  
  ## SETUP STOP CRITERIA ----
  #### 
  ####  NOT FULLY IMPLEMENTED IN THIS FILE.  ONLY FOR STORAGE / VIEWING. 
  ####  ONLY STOPPING CRITERIA HERE IS MAX_EPOCHS.
  ####
  ## test-train split
  
  ttsplit_used <- "test_train" %in% sim_params$stop_criteria || "test_convergence" %in% sim_params$stop_criteria
  ttsplit_ind <- ifelse(
    ttsplit_used,
    floor(sim_params$n_obs * sim_params$ttsplit),
    sim_params$n_obs
  )
  
  if (ttsplit_used){
    x_test <- simdat$x[(ttsplit_ind+1):sim_params$n_obs, ] 
    y_test <- simdat$y[(ttsplit_ind+1):sim_params$n_obs, ]
    loss_test <- torch_zeros(1)  # set initial value
    loss_diff_test <- 1          # set initial value    
  }
  
  x_train <- simdat$x[1:ttsplit_ind, ]
  y_train <- simdat$y[1:ttsplit_ind, ]
  
  ## TRAINING LOOP ----
  ## initialize training params
  loss_diff <- 1
  loss <- torch_zeros(1)
  epoch <- 1
  
  ## stop criteria
  stop_criteria_met <- FALSE
  stop_epochs <- NA
  ## test_mse_storage 
  test_mse_store <- rep(0, times = stop_k + 1)
  test_mse_streak <- rep(FALSE, times = stop_streak)
  
  
  while (!stop_criteria_met){
    prev_loss <- loss
    
    # fit & metrics
    yhat_train <- model_fit(x_train)
    mse <- nnf_mse_loss(yhat_train, y_train)
    kl <- model_fit$get_model_kld() / ttsplit_ind
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
      yhat_test <- model_fit(x_test) 
      # **WOULD LIKE THIS TO BE DETERMINISTIC, i.e. based on post pred means** ----
      mse_test <- nnf_mse_loss(yhat_test, y_test)
      test_mse_store <- roll_vec(test_mse_store, as_array(mse_test))
    }
    
    
    # check test_mse stop criteria:
    # abs(diff(mse)) < sd(mse) for streak epochs in a row,
    # sd calculated using last stop_k epochs
    if (epoch > burn_in & ttsplit_used){
      test_mse_sd <- sd(test_mse_store[1:stop_k])
      test_mse_absdiff <- abs(diff(test_mse_store[stop_k + 0:1]))
      test_mse_compare <- test_mse_absdiff < test_mse_sd
      test_mse_streak <- roll_vec(test_mse_streak, test_mse_compare)
      test_mse_stopcrit <- all(test_mse_store)
      
      if (test_mse_stopcrit){
        stop_epochs <- c(stop_epochs, epoch)
        stop_msg <- paste0(
          "\n \n \n ********************************************* \n",
          "test_mse STOP CONDITION reached at epochs: ", stop_epochs,
          "\n ********************************************* \n \n \n",
        )
        cat_color(stop_msg)
      }
    }

    
    # store results (every `report_every` epochs) ----
    time_to_report <- epoch!=0 & (epoch %% report_every == 0)
    if (time_to_report){
      row_ind <- epoch %/% report_every
      
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
        as_array(dropout_alphas) <= display_alpha_thresh,
        round(as_array(dropout_alphas), 3),
        "."
      )
      cat("alphas below ", round(display_alpha_thresh, 4), ": ")
      cat_color(display_alphas, sep = " ")
      cat(" \n \n")
      
      
      # graphical training updates
      if (want_plots & row_ind > 5){
        
        # only show most recent (for scale of plot)
        start_plot_row_ind <- 1
        if (row_ind >= 20){
          start_plot_row_ind <- row_ind - 10
        } else if (row_ind >= 60){
          start_plot_row_ind <- row_ind - 30
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
      print(plt)
    } # end function update plots (want_fcn_plots = TRUE)
    
    
    # increment epoch counter
    epoch <- epoch + 1
    stop_criteria_met <- epoch > train_epochs
  } # end training WHILE loop
  
  # compile results ----
  sim_res <- list(
    "sim_ind" = sim_ind,
    "stop_epochs" = stop_epochs,
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
  
  # save torch model
  if (save_mod){
    if(is.null(save_mod_path)){
      save_mod_path <- here::here("sims", 
                                  "results", 
                                  paste0("fcnl_hshoe_mod_", 
                                         sim_params$n_obs, "obs_", 
                                         sim_params$sim_seeds[sim_ind],
                                         ".pt"))
    }
    torch_save(model_fit, path = save_mod_path)
    cat_color(txt = paste0("model saved: ", save_mod_path))
    sim_res$mod_path = save_mod_path
  }
  return(sim_res)
}