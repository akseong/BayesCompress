##################################################
## Project:   sparseVCBART data generation & plotting fcns
## Date:      Feb 17, 2026
## Author:    Arnie Seong
##################################################

library(MASS)

# EFFECT MODIFIER FCNS----
beta_0 <- function(Z){
  z1 <- Z[, 1]
  z2 <- Z[, 2]
  3*z1 + 
    (sin(pi * z1)) * (2 - 5 * (z2 > 0.5)) - 
    2 * (z2 > 0.5)
}

# beta_1 Incorrect in sparseVCBART paper ---
#   - plots don't match formula
#   - changed 
# beta_1 <- function(Z){
#   # below is the (incorrect) formula from sparseVCBART paper
#   # plotting this function doesn't match their plot at all
#   # https://arxiv.org/pdf/2510.08204
#   # plot clearly has some type of periodic function in it
#   z1 <- Z[, 1]
#   (3 - 3*z1^2)*(z1 > 0.6) - 10*sqrt(z1)*(z1 < 0.25)
# }

beta_1 <- function(Z){
  # this is function beta_2(z) from the OG VCBART paper
  # https://arxiv.org/pdf/2003.06416
  z1 <- Z[, 1]
  (3 - 3*(z1^2)*cos(6*pi*z1))*(z1 > 0.6) - 10*sqrt(z1)*(z1 < 0.25)
}

beta_2 <- function(Z) {1}

beta_3 <- function(Z){
  # matches beta_4(z) from OG paper
  z1 <- Z[, 1]
  z2 <- Z[, 2]
  z3 <- Z[, 3]
  z4 <- Z[, 4]
  z5 <- Z[, 5]
  10*sin(pi*z1*z2) + 20*((z3-0.5)^2) + 10*z4 + 5*z5
}



## MAKE X VARS FCNS ----
corr_fcn <- function(i, j) {0.5^(abs(i-j))}

make_Covmat <- function(p, covar_fcn){
  Sigma <- matrix(NA, nrow = p, ncol = p)
  for (i in 1:p){
    for (j in 1:p){
      Sigma[i, j] <- covar_fcn(i, j)
    }
  }
  return(Sigma)
}


# GENERATE MEAN FUNCTION DATA----
gen_Eydat_sparseVCBART <- function(
    n_obs = 1e3,
    p = 50,
    R = 20,
    covar_fcn = corr_fcn,
    beta_0 = beta_0,
    beta_1 = beta_1,
    beta_2 = beta_2,
    beta_3 = beta_3
){
  # in sparseVCBART paper, Z1:Z5 important, X1:X3 important
  Covmat <- make_Covmat(p, covar_fcn = corr_fcn)
  X <- as.data.frame(
    mvrnorm(n = n_obs, mu = rep(0, p), Sigma = Covmat)
  )
  names(X) <- paste0("x", 1:p)
  
  ## make Z vars (effect modifiers)
  Z <- as.data.frame(
    matrix(
      runif(R*n_obs, 0, 1),
      nrow = n_obs,
      ncol = R
    )
  )
  names(Z) <- paste0("z", 1:R)
  
  ## apply fcns
  b0 <- beta_0(Z)
  b1 <- beta_1(Z)
  b2 <- beta_2(Z)
  b3 <- beta_3(Z)
  
  # generate response vector
  Ey <- b0 + rowSums(cbind(b1, b2, b3) * X[, 1:3])
  
  true_covs <- c(
    paste0("x", 1:3),
    paste0("z", 1:5)
  )
  
  df <- as.data.frame(cbind(Ey, Z, X))
  return(df)
}


# PLOTTING TRUE BETA FCNS----
# make grids for plotting
plot_b0_true <- function(resol = 100, b0 = beta_0){
  require(tidyverse)
  require(latex2exp)
  z_gridvec <- 0:resol/resol
  z11_plotmat <- cbind(z_gridvec, 1)
  z10_plotmat <- cbind(z_gridvec, 0)
  b0z1_1 <- b0(z11_plotmat)
  b0z1_0 <- b0(z10_plotmat)
  
  plt <- data.frame(
    "z2_0" = b0z1_0,
    "z2_1" = b0z1_1,
    "z1" = z_gridvec
  ) %>% 
    pivot_longer(cols = -3, names_to = "z2val") %>% 
    ggplot() + 
    geom_line(
      aes(y = value, x = z1, color = z2val)
    ) +
    labs(
      title = TeX("$\\beta_0(z)$ ~ $z_1$"),
      subtitle = TeX("fcn differs for $z_2 < 0.5$ and $z_2 > 0.5$")
    )
  
  return(plt)
}


plot_b1_true <- function(resol = 100, b1 = beta_1){
  require(tidyverse)
  require(latex2exp)
  z_gridvec <- 0:resol/resol
  z11_plotmat <- cbind(z_gridvec, 1)
  b1 <- beta_1(z11_plotmat)
  
  plt <- data.frame(
    "b1" = b1,
    "z1" = z_gridvec
  ) %>% 
    ggplot() + 
    geom_line(
      aes(y = b1, x = z1)
    ) +
    labs(
      title = TeX("$\\beta_1(z)$ ~ $z_1$")
    )
  
  return(plt)
}



# MAKE PRED PLOT DFs----
## USE WITH VC MOD NN ----
## make Z grid
make_isolated_Zgrids <- function(R, resol = 100, z2vals = c(0.25, 0.75)){
  zvals <- 1:resol / resol
  zmat <- matrix(0, nrow = R*resol, ncol = R)
  colnames(zmat) <- paste0("z", 1:R)
  
  for (r in 1:R){
    fillrows <- (r-1)*resol + 1:resol
    zmat[fillrows, r] <- zvals
  }
  zmat[1:resol, 2] <- z2vals[1]
  
  # need two vals of z2 for each z1
  z1extra <- zmat[1:resol, ]
  z1extra[, 2] <- z2vals[2]
  
  return(rbind(z1extra, zmat))
}


## USE WITH VANILLA NN ----
# fcns make dataframes used to plot predictions
make_b0_pred_df <- function(
    resol = 100, 
    p = 50,
    R = 20,
    z2_vals = c(.25, .75),
    froth = FALSE,
    froth_mu = 0.25,
    froth_sig = 0.05
){
  # use this to plot Ey ~ z1, for values of z2 below and above 0.5
  # b0(z1, z2) is an intercept term
  # use froth = TRUE to add some noise to the nuisance covars
  b0_z1 <- rep(1:resol/resol, length(z2_vals))
  b0_z2 <- rep(z2_vals, each = resol)
  if (froth){
    zfill <- matrix(
      rnorm((R-2)*length(b0_z1), froth_mu, froth_sig),
      ncol = R-2
    )
  } else {
    z_fill <- matrix(0, nrow = length(b0_z1), ncol = R-2)
  }
  b0_Z <- cbind(b0_z1, b0_z2, z_fill)    
  colnames(b0_Z) <- paste0("z", 1:R)
  
  if (froth){
    zero_mat <- matrix(
      rnorm(p*length(b0_z1), froth_mu, froth_sig),
      ncol = p
    )
  } else {
    zero_mat <- matrix(0, nrow = length(b0_z1), ncol = p)    
  }
  
  colnames(zero_mat) <- paste0("x", 1:p)
  df <- as.data.frame(cbind(b0_Z, zero_mat))
  return(df)
}

make_b1_pred_df <- function(
    resol = 100,
    x1_vals = 1,
    p = 50,
    R = 20,
    froth = FALSE,
    froth_mu = 0.25,
    froth_sig = 0.05
){
  # Use this to plot Ey/x1 against z1 for fixed values of x1
  # Beta_1 only depends on z1.
  # Need to subtract an intercept term (generate using
  #   make_b0_pred_df, set z2vals = 0)
  # use froth = TRUE to add some noise to the nuisance covars
  # 
  b1_z1 <- rep(1:resol/resol, length(x1_vals))
  b1_x1 <- rep(x1_vals, each = resol)
  
  if (froth){
    z_fill <- matrix(
      rnorm((R-1) * length(b1_z1), froth_mu, froth_sig),
      ncol = R-1
    )
    x_fill <- matrix(
      rnorm((p-1) * length(b1_z1), froth_mu, froth_sig),
      ncol = p-1
    )
  } else {
    z_fill <- matrix(0, nrow = length(b1_z1), ncol = R-1)
    x_fill <- matrix(0, nrow = length(b1_z1), ncol = p-1)
  }
  b1_dat <- cbind(b1_z1, z_fill, b1_x1, x_fill)
  colnames(b1_dat) <- c(paste0("z", 1:R), paste0("x", 1:p))
  df <- as.data.frame(b1_dat)
  return(df)
}



make_b2_pred_df <- function(
    resol = 100,
    p = 50,
    R = 20,
    froth = FALSE,
    froth_mu = 0.25,
    froth_sig = 0.05
){
  # Use this to plot Ey/x2 ~ x2
  # beta_2 = 1 (i.e. does not depend on any z)
  b2_x2 <- rep(1:resol/resol)
  
  if (froth){
    z_fill <- matrix(
      rnorm(R * length(b2_x2), froth_mu, froth_sig),
      ncol = R
    )
    x_fill <- matrix(
      rnorm((p-1) * length(b2_x2), froth_mu, froth_sig),
      ncol = p-1
    )
  } else {
    z_fill <- matrix(0, nrow = length(b2_x2), ncol = R)
    x_fill <- matrix(0, nrow = length(b2_x2), ncol = p-1)
  }

  b2_dat <- cbind(
    z_fill, 
    x_fill[, 1], 
    b2_x2, 
    x_fill[, 2:(p-1)]
    )
  colnames(b2_dat) <- c(paste0("z", 1:R), paste0("x", 1:p))
  df <- as.data.frame(b2_dat)
  return(df)
}


# # TESTING----
# 
# ## data generation----
# n_obs <- 1e3
# p <- 50
# R <- 20
# sig_eps <- 1
# mu_eps <- 0
# 
# bfcns_list <- list(
#   "beta_0" = beta_0,
#   "beta_1" = beta_1,
#   "beta_2" = beta_2,
#   "beta_3" = beta_3
# )
# 
# Ey_df <- gen_Eydat_sparseVCBART(
#   n_obs,
#   p,
#   R,
#   covar_fcn = corr_fcn,
#   beta_0 = bfcns_list$beta_0,
#   beta_1 = bfcns_list$beta_1,
#   beta_2 = bfcns_list$beta_2,
#   beta_3 = bfcns_list$beta_3
# )
# 
# range(Ey_df[, 1])
# head(Ey_df)
# true_covs <- c(
#   paste0("x", 1:3),
#   paste0("z", 1:5)
# )
# 
# # param counts
# source(here::here("Rcode", "analysis_fcns.R"))
# param_counts_from_dims(dim_vec = c(R + p, 4, 16, 1))
# 
# 
# plot_b0_true()
# plot_b1_true()
# 
# 
# 
# ## check plotting strat for b0 ----
# b0_pred_df <- make_b0_pred_df()
# head(b0_pred_df)
# 
# b0 <- beta_0(b0_pred_df[, 1:R])
# plot(b0 ~ b0_pred_df[,1])
# 
# b1 <- beta_1(b0_pred_df[, 1:R])
# b2 <- beta_2(b0_pred_df[, 1:R])
# b3 <- beta_3(b0_pred_df[, 1:R])
# 
# Ey_b0 <- b0 + rowSums(cbind(b1, b2, b3) * b0_pred_df[, 1:p + R])
# plot(Ey_b0 ~ b0_pred_df[, 1])
# 
# 
# ## check plotting strat for b1 ----
# # Ey/x1 ~ z1
# # beta_3 = 10*sin(pi*z1*z2) + 20*((z3-0.5)^2) + 10*z4 + 5*z5
# #   so to recover b1, set z2 = 0, z3 = .5, z4=0, z5=0,
# #   or just x3 = 0
# # beta_2 = 1, so just set x2 = 0
# # beta_0 =   3*z1   +   (sin(pi * z1)) * (2 - 5 * (z2 > 0.5))   -   2 * (z2 > 0.5)
# #   also depends on z1; does not depend on any x value,
# #   i.e. cannot separate beta_1 from beta_0
# # So need to subtract an "intercept" from y
# #    - i.e. generate predictions for the same Z values with all x_j = 0
# 
# b1_pred_df <- make_b1_pred_df()
# head(b1_pred_df)
# 
# b1 <- beta_1(b1_pred_df[, 1:R])
# plot(b1 ~ b1_pred_df[,1])
# 
# b0 <- beta_0(b1_pred_df[, 1:R])
# b2 <- beta_2(b1_pred_df[, 1:R])
# b3 <- beta_3(b1_pred_df[, 1:R])
# 
# Ey_b1 <- b0 + rowSums(cbind(b1, b2, b3) * b1_pred_df[, 1:3 + R])
# 
# # without subtracting intercept
# # for x1 = 0, should just be the intercept term
# x1 <- b1_pred_df[,(R+1)]
# df <- data.frame(
#   # "Ey" = Ey,
#   "b1resp" = ifelse(x1 == 0, Ey, Ey/x1),
#   "x1" = x1,
#   "z1" = b1_pred_df[, 1]
# )
# 
# df %>%
#   ggplot() +
#   geom_line(
#     aes(
#       y = b1resp,
#       x = z1,
#       color = as_factor(x1)
#     )
#   )
# 
# 
# ### now subtract intercept predictions
# x1_vals = c(-2, 1, 0, 1, 2)
# b1_pred_df <- make_b1_pred_df(
#   resol = 100,
#   x1_vals = x1_vals,
#   p = 50,
#   R = 20
# )
# 
# b0_for_b1 <- make_b0_pred_df(
#   resol = 100, 
#   p = 50,
#   R = 20,
#   z2_vals = rep(0, length(x1_vals))
# )
# 
# # NOTE: should be able to input X portion of these dataframes
# # into nn_mod instead of what I'm doing here to test
# 
# # generate Ey_b1
# b0 <- beta_0(b1_pred_df[, 1:R])
# b1 <- beta_1(b1_pred_df[, 1:R])
# b2 <- beta_2(b1_pred_df[, 1:R])
# b3 <- beta_3(b1_pred_df[, 1:R])
# Ey_b1 <- b0 + rowSums(cbind(b1, b2, b3) * b1_pred_df[, 1:3 + R])
# 
# # generate intercept term
# b0_int <- beta_0(b0_for_b1[, 1:R])
# b1_int <- beta_1(b0_for_b1[, 1:R])
# b2_int <- beta_2(b0_for_b1[, 1:R])
# b3_int <- beta_3(b0_for_b1[, 1:R])
# int <- b0_int + rowSums(cbind(b1_int, b2_int, b3_int) * b0_for_b1[, 1:3 + R])
# 
# x1 <- b1_pred_df[, R+1]
# z1 <- b1_pred_df[, 1]
# Ey_minus_b0 <- Ey_b1 - int
# plot(Ey_minus_b0 ~ z1)
# 
# b1_to_plot <- Ey_b1/x1 - int 
# df <- as.data.frame(b1_to_plot, Ey_minus_b0, x1, z1) 
# df %>% 
#   ggplot() + 
#   geom_line(
#     aes(
#       y = b1_to_plot,
#       x = z1,
#       color = as_factor(x1)
#     )
#   ) + 
#   labs(
#     title = "Ey/x1 minus intercept ~ z1",
#     subtitle = "x1 = 1 (turqouise) is what should match paper"
#   )
# 
# 
# 
# ## check plotting strat for b2 ----
# # Ey/x2 ~ x2 seems fine
# b2_pred_df <- make_b2_pred_df()
# head(b2_pred_df)
# 
# b2 <- beta_2(b2_pred_df[, 1:R])
# plot(b2 ~ b2_pred_df[,1])
# 
# b0 <- beta_0(b2_pred_df[, 1:R])
# b1 <- beta_1(b2_pred_df[, 1:R])
# b3 <- beta_3(b2_pred_df[, 1:R])
# 
# Ey <- b0 + rowSums(cbind(b1, b2, b3) * b2_pred_df[, 1:p + R])
# plot(Ey/b2_pred_df[, R+2] ~ b2_pred_df[, R+2])


# SIMULATION FCN ----
spVCBART_vanilla_sim <- function(
    sim_params,
    sim_ind,
    sim_save_path,
    nn_model,
    Ey_df, eps_mat
){
  
  ## train/test ----
  # Note: (test obs aren't used to calibrate NN,
  # just for me to observe for training progress
  # and catch simulation problems)
  tr_inds <- 1:sim_params$n_obs
  te_inds <- (sim_params$n_obs + 1):nrow(Ey_df)
  Ey_raw <- Ey_df[, 1]
  XZ_raw <- Ey_df[, -1]
  
  # standardize XZ.  
  # standardizing Y will have to happen after epsilons added
  XZ_means <- colMeans(XZ_raw)
  XZ_sds <- apply(XZ_raw, 2, sd)
  XZ_centered <- sweep(XZ_raw, 2, STATS = XZ_means, "-")
  XZ <- sweep(XZ_centered, 2, STATS = XZ_sds, "/")
  
  XZ_tr <- torch_tensor(as.matrix(XZ[tr_inds, ]))
  XZ_te <- torch_tensor(as.matrix(XZ[te_inds, ]))
  
  ## add noise and standardize y, test/train split ----
  y_raw <- Ey_raw + eps_mat[, sim_ind]
  y_mean <- mean(y_raw)
  y_sd <- sd(y_raw)
  y <- (y_raw - y_mean) / y_sd
  
  y_tr <- torch_tensor(y[tr_inds])$unsqueeze(2)
  y_te <- torch_tensor(y[te_inds])$unsqueeze(2)
  
  if (sim_params$use_cuda){
    y_tr <- y_tr$to(device = "cuda")
    y_te <- y_te$to(device = "cuda")
    XZ_tr <- XZ_tr$to(device = "cuda")
    XZ_te <- XZ_te$to(device = "cuda")
  }
  
  ## when to report / plot ----
  report_epochs <- seq(
    sim_params$report_every, 
    sim_params$train_epochs, 
    by = sim_params$report_every
  )
  
  plot_epochs <- seq(
    sim_params$report_every*sim_params$plot_every_x_reports, 
    sim_params$train_epochs, 
    by = sim_params$report_every*sim_params$plot_every_x_reports
  )
  
  ## store: # train, test mse and kl ----
  loss_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = 3
  )
  colnames(loss_mat) <- c("kl", "mse_train", "mse_test")
  rownames(loss_mat) <- report_epochs
  
  
  ## store: alphas, kappas ----
  alpha_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = sim_params$d_0
  )
  rownames(alpha_mat) <- report_epochs
  
  kappa_local_mat <- 
    kappa_mat <- 
    kappa_tc_mat <-
    kappa_fc_mat <- alpha_mat
  
  
  # TRAIN ----
  ## initialize BNN & optimizer ----
  model_fit <- nn_model()
  optim_model_fit <- optim_adam(model_fit$parameters, lr = sim_params$lr)
  
  ## TRAIN LOOP
  epoch <- 1
  loss <- torch_tensor(1, device = dev_select(sim_params$use_cuda))
  while (epoch <= sim_params$train_epochs){
    
    ## fit & metrics ----
    yhat_tr <- model_fit(XZ_tr)
    mse <- nnf_mse_loss(yhat_tr, y_tr)
    kl <- model_fit$get_model_kld() / length(y_tr)
    loss <- mse + kl
    
    # gradient step 
    # zero out previous gradients
    optim_model_fit$zero_grad()
    # backprop
    loss$backward()
    # update weights
    optim_model_fit$step()
    
    
    ## REPORTING ----
    # track progress
    time_to_report <- epoch %in% report_epochs
    if (!time_to_report & sim_params$verbose){
      if (epoch %% sim_params$report_every == 1){
        cat("Training till next report:")
      }
      # progress bar
      if (sim_params$report_every <= 100){
        # place "." every epoch if report_every < 100
        cat(".")
      } else if (epoch %% round(sim_params$report_every/100) == 1){
        # place "." every percent progress between reports
        cat(".")
      }
    }
    
    ### store results ----
    if (time_to_report){
      row_ind <- epoch %/% sim_params$report_every
      
      # compute test loss
      yhat_te <- model_fit(XZ_te)
      mse_te <- nnf_mse_loss(yhat_te, y_te)
      
      # store params
      loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_te$item())
      dropout_alphas <- model_fit$fc1$get_dropout_rates()
      alpha_mat[row_ind, ] <- as_array(dropout_alphas)
      kappas <- get_kappas(model_fit$fc1)
      kappa_mat[row_ind, ] <- kappas
      
      kappas_local <- get_kappas(model_fit$fc1, type = "local")
      kappa_local_mat[row_ind, ] <- kappas_local
      
      # corrected param kappas
      kappas_tc <- get_kappas_taucorrected(model_fit)
      kappas_fc <- get_kappas_frobcorrected(model_fit)
      kappa_tc_mat[row_ind, ] <- kappas_tc
      kappa_fc_mat[row_ind, ] <- kappas_fc
    }
    
    ### in-console reporting ----
    if (time_to_report & sim_params$verbose){
      cat(
        "\n Epoch:", epoch,
        "MSE + KL/n =", round(mse$item(), 5), "+", round(kl$item(), 5),
        "=", round(loss$item(), 4),
        "\n",
        "train mse:", round(mse$item(), 4), 
        "; test_mse:", round(mse_te$item(), 4),
        sep = " "
      )
      
      # report global shrinkage params (s^2 or tau^2)
      s_sq1 <- get_s_sq(model_fit$fc1)
      s_sq2 <- get_s_sq(model_fit$fc2)
      s_sq3 <- get_s_sq(model_fit$fc3)
      
      cat(
        "\n s_sq1 = ", round(s_sq1, 5),
        "; s_sq2 = ", round(s_sq2, 5),
        "; s_sq3 = ", round(s_sq3, 5),
        sep = ""
      )
      
      # report variable importance indicators
      cat("\n ALPHAS: ", round(as_array(dropout_alphas), 2), "\n")
      # display_alphas <- ifelse(
      #   as_array(dropout_alphas) <= sim_params$alpha_thresh,
      #   round(as_array(dropout_alphas), 3),
      #   "."
      # )
      # cat("alphas below 0.82: ")
      # cat(display_alphas, sep = " ")
      cat("\n LOCAL kappas: ", round(kappas_local, 2), "\n")
      cat("\n GLOBAL kappas: ", round(kappas, 2), "\n")
      cat("\n FROB kappas: ", round(kappas_fc, 2), "\n")
      cat("\n TAU kappas: ", round(kappas_tc, 2), "\n")
      cat("\n \n")
    }
    
    ## PLOTS ----
    time_to_plot <- epoch %in% plot_epochs
    ### metric plot ----
    if (time_to_plot & sim_params$want_metric_plts){
      # plot train/test MSE, KL
      metrics_plt <- varmat_pltfcn(
        tail(loss_mat, 50), 
        show_vars = colnames(loss_mat)
      )$show_vars_plt + 
        labs(title = "training metrics vs epoch")
      
      # save or print
      if (sim_params$save_fcn_plts){
        loss_plt_fname <- paste0(sim_save_path, "_loss_e", round(epoch/1000), "k.png")
        ggsave(filename = loss_plt_fname, plot = metrics_plt, height = 4, width = 6)
      } else {
        print(metrics_plt)
      }
      
    }
    
    ### fcn plots ----
    if (time_to_plot & sim_params$want_fcn_plts){
      
      subtitle_str <- paste0("epoch :", epoch)
      
      ## plot beta_0 
      b0_df_raw <- make_b0_pred_df(
        p = sim_params$p, 
        R = sim_params$R,
        z2_vals = c(0,1)
      )
      b0_df <- scale_mat(b0_df_raw, means = XZ_means, sds = XZ_sds)$scaled
      b0_yhat <- as_array(
        get_nn_mod_Ey(
          nn_mod = model_fit, 
          X = torch_tensor(as.matrix(b0_df))
        )
      )
      b0_yhat_raw <- b0_yhat * y_sd + y_mean
      
      b0_plt <- data.frame(
        "b0" = bfcns_list$beta_0(b0_df_raw),
        "b0_hat" = b0_yhat_raw,
        "z1" = b0_df_raw[, 1],
        "z2" = as_factor(b0_df_raw[, 2])
      ) %>% 
        pivot_longer(cols = 1:2, values_to = "b0") %>% 
        ggplot(aes(y = b0, x = z1, color = z2, linetype = name)) + 
        geom_line() +
        labs(
          title = TeX("estimated $\\beta_0$ ~ $z_1$"),
          subtitle = subtitle_str
        )
      
      
      ## plot beta_1
      b1_df_raw <- make_b1_pred_df(
        resol = 100,
        x1_vals = 1,
        p = sim_params$p,
        R = sim_params$R
      )
      b1_df <- scale_mat(b1_df_raw, XZ_means, XZ_sds)$scaled
      b0_for_b1_raw <- make_b0_pred_df(
        resol = 100,
        p = sim_params$p,
        R = sim_params$R,
        z2_vals = 0
      )
      b0_for_b1_df <- scale_mat(b0_for_b1_raw, XZ_means, XZ_sds)$scaled
      
      # gen scaled Eys
      Ey_b1_hat <- as_array(
        get_nn_mod_Ey(
          nn_mod = model_fit, 
          X = torch_tensor(as.matrix(b1_df))
        )
      )
      b0_hat <- as_array(
        get_nn_mod_Ey(
          nn_mod = model_fit, 
          X = torch_tensor(as.matrix(b0_for_b1_df))
        )
      )
      
      # indirectly get b1:
      b1_hat <- Ey_b1_hat/b1_df$x1 - b0_hat
      b1_hat_to_plot <- (b1_hat*y_sd) + y_mean
      
      b1_pltdf <- data.frame(
        "b1_true" = bfcns_list$beta_1(b1_df_raw),
        "b1_hat" = b1_hat_to_plot,
        "z1" = b1_df_raw$z1,
        "x1" = b1_df_raw$x1
      )
      
      b1_plt <- b1_pltdf %>%
        pivot_longer(cols = 1:2) %>% 
        ggplot() +
        geom_line(
          aes(
            y = value,
            x = z1,
            color = name
          )
        ) +
        labs(
          title = "Ey_hat/x1 minus b0_hat ~ z1",
          subtitle = subtitle_str
        )
      
      # save or print
      if (sim_params$save_fcn_plts){
        b0_plt_fname <- paste0(sim_save_path, "_b0_e", round(epoch/1000), "k.png")
        b1_plt_fname <- paste0(sim_save_path, "_b1_e", round(epoch/1000), "k.png")
        ggsave(filename = b0_plt_fname, plot = b0_plt, height = 4, width = 6)
        ggsave(filename = b1_plt_fname, plot = b1_plt, height = 4, width = 6)
      } else {
        print(b0_plt)
        print(b1_plt)
      }
      
    }
    
    # increment
    epoch <- epoch + 1
  }
  
  ### compile results ----
  sim_res <- list(
    "sim_ind" = sim_ind,
    "sim_params" = sim_params,
    "loss_mat" = loss_mat,
    "alpha_mat" = alpha_mat,
    "kappa_mat" = kappa_mat,
    "kappa_tc_mat" = kappa_tc_mat,
    "kappa_fc_mat" = kappa_fc_mat,
    "kappa_local_mat" = kappa_local_mat
  )
  
  ### notify completed training ----
  completed_msg <- paste0(
    "\n \n ******************** \n ******************** \n",
    "sim #", 
    sim_ind, 
    " completed",
    "\n ******************** \n ******************** \n \n"
  )
  cat_color(txt = completed_msg)
  
  ### save torch model & sim results ----
  if (sim_params$save_mod){
    save_mod_path <- paste0(sim_save_path, ".pt")
    torch_save(model_fit, path = save_mod_path)
    cat_color(txt = paste0("model saved: ", save_mod_path))
    sim_res$mod_path = save_mod_path
  }
  
  if (sim_params$save_results){
    save_res_path <- paste0(sim_save_path, ".RData")
    save(sim_res, file = save_res_path)
    cat_color(txt = paste0("sim results saved: ", save_res_path))
  }
  
  return(sim_res)
}
