



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
XZ_noint <- sweep(XZ_centered, 2, STATS = XZ_sds, "/")

# add intercept!
XZ <- cbind(
  XZ_noint[, 1:sim_params$R], 
  1, 
  XZ_noint[, (1 + sim_params$R):(sim_params$R + sim_params$p)]
)

XZ_tr <- torch_tensor(as.matrix(XZ[tr_inds, ]))
XZ_te <- torch_tensor(as.matrix(XZ[te_inds, ]))

## add noise and standardize y, test/train split ----
y_raw <- Ey_raw + eps_mat[, sim_ind]
y_mean <- mean(y_raw)
y_sd <- sd(y_raw)
y <- (y_raw - y_mean) / y_sd

y_tr <- torch_tensor(y[tr_inds])
y_te <- torch_tensor(y[te_inds])

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
  ncol = sim_params$d_0 + sim_params$d_p1
)
rownames(alpha_mat) <- report_epochs
colnames(alpha_mat) <- c(
  paste0("z", 1:sim_params$R),
  "b_int",
  paste0("b", 1:sim_params$p)
)


kappa_local_mat <- 
  kappa_mat <- 
  kappa_tc_mat <- alpha_mat
  # kappa_fc_mat <- alpha_mat


# TRAIN ----
## initialize BNN & optimizer ----
model_fit <- nn_model()
optim_model_fit <- optim_adam(model_fit$parameters, lr = sim_params$lr)

## TRAIN LOOP
epoch <- 1
loss <- torch_tensor(1, device = dev_select(sim_params$use_cuda))
while (epoch <= sim_params$train_epochs){
  
  ## fit & metrics ----
  yhat_tr <- model_fit(
    zvars = XZ_tr[, 1:sim_params$d_0], 
    xvars = XZ_tr[, (1 + sim_params$d_0):(sim_params$d_0 + sim_params$d_p1)]
  )

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
    yhat_te <- model_fit(
      zvars = XZ_te[, 1:sim_params$d_0], 
      xvars = XZ_te[, (1 + sim_params$d_0):(sim_params$d_0 + sim_params$d_p1)]
    )
    mse_te <- nnf_mse_loss(yhat_te, y_te)
    loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_te$item())
    
    # store params
    dropout_alphas <- c(
      as_array(model_fit$fc1$get_dropout_rates()),
      as_array(model_fit$vc$get_dropout_rates())
    )
    kappas <- c(
      get_kappas(model_fit$fc1),
      get_kappas(model_fit$vc)
    )
    kappas_local <- c(
      get_kappas(model_fit$fc1, type = "local"),
      get_kappas(model_fit$vc, type = "local")
    )
    kappas_tc <- c(
      get_kappas_taucorrected(model_fit),
      get_kappas(model_fit$vc)
    )
    
    # # omitted frobenius-corrected kappas
    # kappas_fc <- c(
    #   get_kappas_frobcorrected(model_fit),
    #   get_kappas(model_fit$vc)
    # )
    
    alpha_mat[row_ind, ] <- dropout_alphas
    kappa_mat[row_ind, ] <- kappas
    kappa_local_mat[row_ind, ] <- kappas_local
    # kappa_fc_mat[row_ind, ] <- kappas_fc
    kappa_tc_mat[row_ind, ] <- kappas_tc
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
    s_sqvc <- get_s_sq(model_fit$vc)
    
    cat(
      "\n s_sq1 = ", round(s_sq1, 5),
      "; s_sq2 = ", round(s_sq2, 5),
      "; s_sq3 = ", round(s_sq3, 5),
      "; s_sqvc = ", round(s_sqvc, 5),
      sep = ""
    )
    
    # report variable importance indicators
    cat("\n ALPHAS: ", round(dropout_alphas, 2), "\n")
    # display_alphas <- ifelse(
    #   as_array(dropout_alphas) <= sim_params$alpha_thresh,
    #   round(as_array(dropout_alphas), 3),
    #   "."
    # )
    # cat("alphas below 0.82: ")
    # cat(display_alphas, sep = " ")
    cat("\n LOCAL kappas: ", round(kappas_local, 2), "\n")
    cat("\n GLOBAL kappas: ", round(kappas, 2), "\n")
    # cat("\n FROB kappas: ", round(kappas_fc, 2), "\n")
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
    
    ## make Z grid
    make_isolated_Zgrids <- function(R, resol = 100){
      zvals <- 1:resol / resol
      zmat <- matrix(0, nrow = R*resol, ncol = R)
      
      for (r in 1:R){
        fillrows <- (r-1)*resol + 1:resol
        zmat[fillrows, r] <- zvals
      }
      
      # need two vals of z2 for each z1
      z1extra
      
    }
    
    
    b0_df_raw <- make_b0_pred_df(
      p = sim_params$p, 
      R = sim_params$R,
      z2_vals = c(0,1)
    )
    b0_df <- scale_mat(b0_df_raw, means = XZ_means, sds = XZ_sds)$scaled
    b0_df_z <- torch_tensor(as.matrix(b0_df[, 1:sim_params$d_0]))
    b0_df_x <- torch_tensor(as.matrix(
      cbind(1, b0_df[, (1 + sim_params$d_0):(sim_params$d_0 + sim_params$p)])
    ))
    
    b0_yhat <-
      model_fit$forward(
        zvars = b0_df_z, 
        xvars = b0_df_x,
        want_betas = TRUE
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