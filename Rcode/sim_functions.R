##################################################
## Project:   simulation functions
## Date:      Jul 17, 2024
## Author:    Arnie Seong
##################################################


# MISC UTILITIES ----

## %notin% ----
`%notin%` <- Negate(`%in%`)

## time_since ----
time_since <- function(){
  # prints time since last called 
  # (use to print time a loop takes, for example).
  # usage:
  # f <- time_since()
  # f()
  # f()
  
  st <- Sys.time()
  function(x = Sys.time()) {
    print(x-st)
    st <<- x
  }
}

## find_array_ind ----
find_array_ind <- function(array, name, marg = 1){
  # marg = 1 --> rows;    marg = 2 --> cols
  # generalizes to higher-dim arrays
  inds <- which(dimnames(array)[[marg]] == name)
  
  if (length(inds) == 0){
    warning("no corresponding row found")
  } else if (length(inds) > 1) {
    warning("multiple corresponding rows found")
    return(inds)
  } else if (length(inds) == 1) {
    return(inds)
  }
}



## update mat during training
update_matrix_row <- function(mat, epoch, update_vec, reportevery = 100, verbose = FALSE){
  # USAGE EXAMPLE:  
  #    mat <- matrix(NA, nrow = 20, update_vec = rep(1, 5) ncol = 5)
  #    for (i in 1:1000){
  #      mat <- matfunc(mat, epoch = i)
  #    }
  
  if (!epoch%%reportevery){
    row_ind <- epoch%/%reportevery
    mat[row_ind, ] <- update_vec
    if (verbose) {
      cat("row", row_ind, 
          "updated on epoch", epoch, 
          "with", update_vec,
          sep = " ")
    }
  } else if (epoch%%reportevery & verbose){
    cat("no update on epoch ", epoch)
  }
  return(mat)
}





## cat_color(txt, style = 1, color = 36) ---
cat_color <- function(txt, sep_char = ", ", style = 1, color = 36){
  # prints txt with colored font/bkgrnd
  cat(
    paste0(
      "\033[0;",
      style, ";",
      color, "m",
      txt,
      "\033[0m"
    ),
    sep = sep_char
  )  
}




## mathematical operations
softplus <- function(x){
  log(1 + exp(x))
}

sigmoid <- function(x){
  1/(1 + exp(-x))
}

softmax <- function(z){
  exp(z) / sum(exp(z))
}

clamp <- function(v, lo = 0, hi = 1){
  #ensure values are in [0, 1]
  v <- ifelse( v <= hi, v, hi)
  v <- ifelse( v >= lo, v, lo)
  return(v)
}




## vismat ----
vismat <- function(mat, cap = NULL, lims = NULL, leg = TRUE, na0 = TRUE, square){
  # outputs visualization of matrix with few unique values
  # colnames should be strings, values represented as factors
  # sci_not=TRUE puts legend in scientific notation
  require(ggplot2)
  require(scales)
  require(reshape2)
  
  melted <- melt(mat)
  melted$value <- ifelse(
    melted$value == 0 & na0,
    NA,
    melted$value
  )
  p <- ggplot(melted) + 
    geom_raster(aes(y = Var1, 
                    x = Var2, 
                    fill = value)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    scale_fill_viridis_c(limits = lims)
  
  if (is.numeric(melted$Var1)){
    p <- p + 
      scale_y_reverse()
  } else {
    p <- p + 
      scale_y_discrete(limits = rev(levels(melted$Var1)))
  }
  
  if (missing(square)) square <- nrow(mat) / ncol(mat) > .9 & nrow(mat) / ncol(mat) < 1.1
  if (square) p <- p + coord_fixed(1)
  
  if (!is.null(cap)) p <- p + labs(title=cap)
  
  if (!leg) p <- p + theme(legend.position = "none")
  
  return(p)
}
# DATA GENERATION ----
## sim_linear_data ----
sim_linear_data <- function(
  n_obs = 100,
  err_sigma = 1,
  true_coefs = NULL,
  d_in = 10,
  d_true = 3,
  intercept = 0
){
  require(torch)
  
  # if true_coefs not provided, generates randomly
  if (is.null(true_coefs)){
    true_coefs <- round(runif(d_in,-5, 5), 2)
    true_coefs[(d_true + 1): d_in] <- 0
  }
  
  # ensure d_in, d_true match true_coefs (if true_coefs provided)
  d_in <- length(true_coefs)
  d_true <- sum(true_coefs != 0)
  
  # generate x, y
  x <- torch_randn(n_obs, d_in)
  y <- x$matmul(true_coefs)$unsqueeze(2) + 
    intercept + 
    torch_normal(mean = 0, std = err_sigma, size = c(n_obs, 1))
  
  return(
    list(
      "y" = y,
      "x" = x,
      "true_coefs" = true_coefs,
      "intercept" = intercept,
      "n_obs" = n_obs,
      "d_in" = d_in,
      "d_true" = d_true
    )
  )
}


## sim_func_data----

# # sample functions to use
# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x)
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x

# # slightly harder setting
# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) - (abs(x))
# flist = list(fcn1, fcn2, fcn3, fcn4)

sim_func_data <- function(
  n_obs = 1000,
  d_in = 10,
  flist = list(fcn1, fcn2, fcn3, fcn4),
  err_sigma = 1
){
  # generate x, y
  x <- torch_randn(n_obs, d_in)
  y <- rep(0, n_obs)
  for(j in 1:length(flist)){
    y <- y + flist[[j]](x[,j])
  }
  y <- y$unsqueeze(2) + torch_normal(mean = 0, std = err_sigma, size = c(n_obs, 1))
  
  return(
    list(
      "y" = y,
      "x" = x,
      "n_obs" = n_obs,
      "d_in" = d_in,
      "d_true" = length(flist)
    )
  )
}
sim_func_data <- function(
  n_obs = 1000,
  d_in = 10,
  flist = list(fcn1, fcn2, fcn3, fcn4),
  err_sigma = 1
){
  # generate x, y
  x <- torch_randn(n_obs, d_in)
  y <- rep(0, n_obs)
  for(j in 1:length(flist)){
    y <- y + flist[[j]](x[,j])
  }
  y <- y$unsqueeze(2) + torch_normal(mean = 0, std = err_sigma, size = c(n_obs, 1))
  
  return(
    list(
      "y" = y,
      "x" = x,
      "n_obs" = n_obs,
      "d_in" = d_in,
      "d_true" = length(flist)
    )
  )
}




## sim_func_data_unifx----
sim_func_data_unifx <- function(
    n_obs = 1000,
    d_in = 10,
    flist = list(fcn1, fcn2, fcn3, fcn4),
    err_sigma = 1,
    xlo = -5,
    xhi = 5
){
  # simulate covariates X from uniform distributions
  # generate x, y
  x <- (xlo - xhi) * torch_rand(n_obs, d_in) + xhi
  y <- rep(0, n_obs)
  for(j in 1:length(flist)){
    y <- y + flist[[j]](x[,j])
  }
  y <- y$unsqueeze(2) + torch_normal(mean = 0, std = err_sigma, size = c(n_obs, 1))
  
  return(
    list(
      "y" = y,
      "x" = x,
      "n_obs" = n_obs,
      "d_in" = d_in,
      "d_true" = length(flist)
    )
  )
}



# ASSESSMENT FUNCS ----

## binary error----
### binary_err_mat ----
binary_err_mat <- function(est, tru){
  # returns 4-row matrix of FP, TP, FN, TN
  FP <- est - tru == 1
  TP <- est + tru == 2
  FN <- est - tru == -1
  TN <- abs(tru) + abs(est) == 0
  return(rbind(FP, TP, FN, TN))
}

### binary_err----
binary_err <- function(est, tru){
  # returns FP, TP, FN, TN as percentage of all decisions
  rowSums(binary_err_mat(est, tru)) / length(tru)  
}

### binary_err_rate----
binary_err_rate <- function(est, tru){
  # returns FP, TP, FN, TN rates
  decision_counts <- rowSums(binary_err_mat(est, tru))
  actual_pos <- decision_counts[2] + decision_counts[3]
  actual_neg <- decision_counts[1] + decision_counts[4]
  
  denom <- c(actual_neg, actual_pos, actual_pos, actual_neg)
  # in case no positives or negatives predicted
  denom <- ifelse(denom == 0, 1, denom) 
  decision_counts / denom
}

## plotting predicted functions ----

### make_pred_mats----
make_pred_mats <- function(flist, xgrid = seq(-4.9, 5, length.out = 100), d_in){
  require(torch)
  n_truevars <- length(flist)
  # make torch array to pass into torchmod
  x_grids <- torch_zeros(n_truevars*length(xgrid), d_in)
  y_fcn <- rep(NA, n_truevars*length(xgrid))
  for(i in 1:n_truevars){
    st_row <- i*length(xgrid)-(length(xgrid)-1)
    end_row <- i*length(xgrid)
    x_grids[st_row:end_row, i] <- xgrid
    y_fcn[st_row:end_row] <- flist[[i]](xgrid)
  }
  
  vis_df <- data.frame(
    "y_true" = y_fcn,
    "x" = rep(xgrid, n_truevars),
    "name" = rep(paste0("x.", 1:n_truevars), each = length(xgrid))
  ) 
  
  return(
    list(
      "vis_df" = vis_df,
      "x_tensor" = x_grids
    )
  )
}

### plot_fcn_preds----
plot_fcn_preds <- function(torchmod, pred_mats, want_df = FALSE, want_plot = TRUE){
  vis_df <- pred_mats$vis_df
  vis_df$y_pred <- as_array(torchmod(pred_mats$x_tensor))
  plt <- vis_df %>% 
    ggplot() + 
    geom_line(
      aes(y = y_pred, x = x, color = name)
    ) + 
    geom_line(
      aes(
        y = y_true,
        x = x,
        color = name
      ),
      linewidth = 1,
      alpha = 0.15
    ) + 
    labs(
      title = "predicted and true fcns",
      color = ""
    )
  
  if (want_df & want_plot){
    return(list(
      "vis_df" = vis_df,
      "plt" = plt
    ))
  } else if (want_df) {
    return(vis_df)
  } else if (want_plot){
    plt
  }
}




# FOR LM() ----
##calc_lm_stats----
calc_lm_stats <- function(lm_fit, true_coefs, alpha = 0.05){
  beta_hat <- summary(lm_fit)$coef[-1, 1]
  binary_err <- binary_err_rate(
    est = summary(lm_fit)$coef[-1, 4] < alpha, 
    tru = true_coefs != 0)
  fit_mse <- mean(lm_fit$residuals^2)
  coef_mse <- mean((beta_hat - true_coefs)^2)
  list(
    "binary_err" = binary_err,
    "fit_mse" = fit_mse,
    "coef_mse" = coef_mse
  )
}

## get_lm_stats ----
get_lm_stats <- function(simdat, alpha = 0.05){
  lm_df <- data.frame(
    "y" = as_array(simdat$y), 
    "x" = as_array(simdat$x)
  )
  if (simdat$d_in > simdat$n_obs){
    lm_df <- lm_df[, 1:(ceiling(simdat$n_obs/2)+1)]
  }
  
  lm_fit <- lm(y ~ ., lm_df)
  if (length(simdat$true_coefs) >= simdat$n_obs){
    warning("p >= n; (p - n) + floor(n/2) spurious covariates eliminated to accomodate lm")
    calc_lm_stats(
      lm_fit = lm_fit, 
      true_coefs = simdat$true_coefs[1:ceiling(simdat$n_obs/2)], 
      alpha = alpha
    )
  } else {
    calc_lm_stats(lm_fit = lm_fit, true_coefs = simdat$true_coefs, alpha = alpha)
  }
}



# FOR SPIKE-SLAB ----



# FOR MOMBF ----


# FOR LASSO ----



# SIMULATION FCNS ----
## horseshoe, basic linear regression setting ----
sim_fcn_hshoe_linreg <- function(
    sim_ind,    # to ease parallelization
    sim_params,     # see example
    nn_model,   # torch nn_module,
    train_epochs = 10000,
    verbose = TRUE,      # provide updates in console
    report_every = 1000, # training epochs between display/store results
    want_plots = TRUE,   # provide graphical updates of KL, MSE
    want_all_params = FALSE, 
    want_data = FALSE    # include training data in output
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
  
  ## generate linear data
  set.seed(sim_params$sim_seeds[sim_ind])
  torch_manual_seed(sim_params$sim_seeds[sim_ind])
  
  lin_simdat <- sim_linear_data(
    n = sim_params$n_obs,
    true_coefs = sim_params$true_coefs,
    err_sigma = sim_params$err_sig
  )
  
  # store: # train, test mse and kl
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
    x_test <- lin_simdat$x[(ttsplit_ind+1):sim_params$n_obs, ] 
    y_test <- lin_simdat$y[(ttsplit_ind+1):sim_params$n_obs, ]
    loss_test <- torch_zeros(1)  # set initial value
    loss_diff_test <- 1          # set initial value    
  }
  
  x_train <- lin_simdat$x[1:ttsplit_ind, ]
  y_train <- lin_simdat$y[1:ttsplit_ind, ]
  
  ## initialize BNN & optimizer ----
  model_fit <- nn_model()
  optim_slhs <- optim_adam(model_fit$parameters)
  
  ## TRAINING LOOP ----
  ## initialize training params
  loss_diff <- 1
  loss <- torch_zeros(1)
  epoch <- 1
  stop_criteria_met <- FALSE
  
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
        as_array(dropout_alphas) <= .1,
        round(as_array(dropout_alphas), 3),
        "."
      )
      cat("alphas below 0.1: ")
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
    
    # increment epoch counter
    epoch <- epoch + 1
    stop_criteria_met <- epoch > train_epochs
  } # end training WHILE loop
  
  # compile results ----
  sim_res <- list(
    "sim_ind" = sim_ind,
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
  
  if (want_data){
    # add data
    sim_res$x = as_array(lin_simdat$x)
    sim_res$y = as_array(lin_simdat$y)
    sim_res$true_coefs = lin_simdat$true_coefs
  }
  
  completed_msg <- paste0("\n \n ******************** \n sim #", 
                          sim_ind, 
                          " completed \n ",
                          "final alphas: ",
                          paste0(round(as_array(dropout_alphas), 3), collapse = ", "),
                          "\n ********************\n \n")
  cat_color(txt = completed_msg)
  return(sim_res)
}

## horseshoe, functional data ----
# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) - (abs(x))
# flist = list(fcn1, fcn2, fcn3, fcn4)

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
    save_mod_path = NULL
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
  stop_criteria_met <- FALSE
  
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
  
  completed_msg <- paste0("\n \n ******************** \n ******************** \n",
                          "sim #", 
                          sim_ind, 
                          " completed \n ",
                          "final alphas below threshold: ",
                          paste0(display_alphas, collapse = " "),
                          "\n ******************** \n ******************** \n \n")
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
