##################################################
## Project:   Sim analysis
## Date:      Dec 23, 2025
## Author:    Arnie Seong
##################################################

# #### setup ----
# library(here)
# library(tidyr)
# library(dplyr)
# library(ggplot2)
# library(gridExtra)
# 
# library(torch)
# source(here("Rcode", "torch_horseshoe_klcorrected.R"))
# source(here("Rcode", "sim_functions.R"))
# # source(here("Rcode", "sim_hshoe_normedresponse.R"))
# 
# if (torch::cuda_is_available()){
#   use_cuda <- TRUE
#   message("Default tensor device set to GPU (CUDA).")
# } else {
#   use_cuda <- FALSE
#   message("Default tensor device remains CPU.")
# }
# 
# 
# 
# fname_stem <- here::here("sims", "results", "hshoe_smooth_pvtau_1721632_12500obs_")
# load(paste0(fname_stem, "108703.RData"))
# 
# sim_res$loss_mat
# sim_res$mod_path
# sim_res$sim_params$sim_seeds


# PLOTTING over epochs ----

## loss_pltfcn ----
loss_pltfcn <- function(sim_res, report_every = NULL, burn = 5e4){
  if (is.null(report_every)) {
    report_every <- sim_res$sim_params$report_every
  }
  
  long_df <- sim_res$loss_mat %>% 
    as_data_frame() %>% 
    mutate("epoch" = row_number() * report_every) %>% 
    pivot_longer(
      cols = c("kl", "mse_train", "mse_test"),
      names_to = "loss_type"
    )
  
  mse_plt <- long_df %>% 
    filter(epoch > burn) %>% 
    filter(loss_type %in% c("mse_train", "mse_test")) %>% 
    ggplot(
      aes(y = value, x = epoch, color = loss_type)
    ) +
    geom_line() + 
    labs(
      title = paste0("MSE reported every ", report_every / 1000, "k epochs"),
      subtitle = paste0("initial ", burn / 1000, "k epochs omitted for scale")
    )
  
  kl_plt <- long_df %>% 
    filter(epoch > burn) %>% 
    filter(loss_type == "kl") %>% 
    ggplot(
      aes(y = value, x = epoch, color = loss_type)
    ) +
    geom_line() + 
    labs(
      title = paste0("KL reported every ", report_every / 1000, "k epochs"),
      subtitle = paste0("initial ", burn / 1000, "k epochs omitted for scale")
    )
  
  return(
    list(
      "long_df" = long_df,
      "mse_plt" = mse_plt,
      "kl_plt" = kl_plt
    )
  )
}

### usage ----
# loss_pltfcn(sim_res, burn = 0)$mse_plt
# loss_pltfcn(sim_res, burn = 8e5)$mse_plt
# 
# loss_pltfcn(sim_res, burn = 0)$kl_plt  + 
#   theme(legend.position = "none")
# loss_pltfcn(sim_res, burn = 5e5)$kl_plt  + 
#   theme(legend.position = "none")



## varmat_pltfcn ----
varmat_pltfcn <- function(
  varmat, 
  y_name = NULL,
  burn = 0,
  report_every = 1e4,
  show_vars = paste0("x", 1:4)
){
  if (is.null(colnames(varmat))) {
    nvars <- ncol(varmat)
    colnames(varmat) <- paste0("x", 1:nvars)
  }
  
  long_df <- varmat %>% 
    as_data_frame() %>%
    mutate("epoch" = row_number() * report_every) %>% 
    filter(epoch > burn) %>%
    pivot_longer(
      cols = -epoch, 
      names_to = "var",
    )
  
  all_vars_plt <-  ggplot() +
    geom_line(
      data = subset(long_df, var %notin% show_vars),
      aes(y = value, x = epoch, group = var),
      alpha = 0.1,
      show.legend = FALSE
    ) + 
    geom_line(
      data = subset(long_df, var %in% show_vars),
      aes(y = value, x = epoch, color = var)
    )
  
  show_vars_plt <- long_df %>%  
    filter(var %in% show_vars) %>%  
    ggplot(
      aes(y = value, x = epoch, color = var)
    ) +
    geom_line() 
  
  if (!is.null(y_name)){
    title_str <- paste0(y_name, " vs epochs")
    all_vars_plt <- all_vars_plt + labs(title = paste0("All vars: ", title_str), y = y_name)
    show_vars_plt <- show_vars_plt + labs(title = paste0("Selected vars: ", title_str), y = y_name) 
  }
  
  if (burn != 0){
    subtitle_str <- paste0("initial ", burn / 1000, "k epochs omitted for scale")
    all_vars_plt <- all_vars_plt + labs(subtitle = subtitle_str)
    show_vars_plt <- show_vars_plt + labs(subtitle = subtitle_str)
  }
  
  return(
    list(
      "long_df" = long_df,
      "all_vars_plt" = all_vars_plt,
      "show_vars_plt" = show_vars_plt
    )
  )
}

### usage ----
# varmat_pltfcn(
#   sim_res$alpha_mat,
#   y_name = "alpha",
#   burn = 0
# )$all_vars_plt
# 
# varmat_pltfcn(
#   sim_res$alpha_mat,
#   y_name = "alpha",
#   burn = 25e4
# )$show_vars_plt
# 
# 
# varmat_pltfcn(
#   sim_res$kappa_mat,
#   y_name = "GLOBAL kappa",
#   burn = 0
# )$all_vars_plt
# 
# varmat_pltfcn(
#   sim_res$kappa_local_mat,
#   y_name = "LOCAL kappa",
#   burn = 0
# )$all_vars_plt
# 
# varmat_pltfcn(
#   sim_res$kappa_local_mat,
#   y_name = "LOCAL kappa",
#   burn = 25e4
# )$show_vars_plt



# DECISIONS over epoch ---- 

## err_by_epoch ----
err_by_epoch <- function(
  dropout_mat, 
  true_vec = c(rep(1, 4), rep(0, 100)),
  max_bfdr = 0.01
){
  resmat <- t(
    apply(
      dropout_mat, 1, 
      function(X) 
        err_from_dropout(
          dropout_vec = X,
          max_bfdr = max_bfdr,
          true_gam = true_vec
        )
    )
  )
  return(resmat)
}

### usage ----
# err_by_epoch(
#   dropout_mat = sim_res$kappa_local_mat,
#   max_bfdr = 0.01
# )
# 
# err_by_epoch(
#   dropout_mat = sim_res$alpha_mat,
#   max_bfdr = 0.01
# )
# 
# err_by_epoch(
#   dropout_mat = sim_res$kappa_mat,
#   max_bfdr = 0.01
# )
# 
# err_by_epoch(
#   dropout_mat = sim_res$kappa_local_mat,
#   max_bfdr = 0.01
# )












# EXTRACT from `sim_res` fcns ----
## kappamat_from_sim_res ----
kappamat_from_sim_res <- function(sim_res, ln_fcn = ln_mode, type = "global"){
  ztil_mat <- array(NA, dim = dim(sim_res$atilde_mu_mat))
  for (i in 1:nrow(ztil_mat)){
    ztil_mat[i, ] <- ln_fcn(
      mu = sim_res$atilde_mu_mat[i, ] + sim_res$btilde_mu_mat[i, ],
      var = exp(sim_res$atilde_logvar_mat[i, ]) + exp(sim_res$btilde_logvar_mat[i, ])
    )
  }
  
  if (type == "global"){
    s_sq_vec <- ln_fcn(
      mu = sim_res$sa_mu_vec + sim_res$sb_mu_vec,
      var = exp(sim_res$sa_logvar_vec) + exp(sim_res$sb_logvar_vec)
    )
    # test sweep
    # tmat <- matrix(1:9, nrow = 3)
    # tvec <- 1:3
    # sweep(tmat, 1, tvec, FUN = "*")
    zmat <- sweep(ztil_mat, 1, s_sq_vec, FUN = "*")
    return((1 + zmat)^(-1))
  } else if (type == "local"){
    return((1 + ztil_mat)^(-1))
  } else {
    warn("type must be 'global' or 'local'")
  }
}









# ERR calculations ----

## err_by_max_bfdr ----
err_by_max_bfdr <- function(
    dropout_vec, 
    true_vec = c(rep(1, 4), rep(0, 100)), 
    bfdr_vec = 1:99/100
){
  
  err_mat <- matrix(NA, ncol = 6, nrow = length(bfdr_vec))
  for (i in 1:length(bfdr_vec)){
    err_mat[i, ] <- err_from_dropout(
      dropout_vec,
      true_gam = true_vec,
      max_bfdr = bfdr_vec[i]
    )
  }
  res_mat <- cbind(bfdr_vec, err_mat)
  colnames(res_mat) <- c("max_bfdr", "fdr", "bfdr", "FP", "TP", "FN", "TN")
  
  plt <- res_mat %>% 
    as_data_frame() %>% 
    pivot_longer(cols = 4:5) %>% 
    ggplot(aes(y = value, x = max_bfdr, color = name)) + 
    geom_line() + 
    labs(
      title = "TPR & FPR ~ max_bfdr"
    )
  
  return(
    list(
      "err_mat" = res_mat,
      "plt" = plt
    )
  )
}




# CORRECTIONS ----

## m_eff ----
m_eff <- function(nn_layer){
  k <- get_kappas(nn_layer)
  sum(1 - k)
}





# OTHER ----
# none are great
# alphas are stable and work well if BFDR is very low.  
# global kappas don't want to selet anything
# local kappas 



# something happens around 300k epochs in this simulation;
# KL has large drop
# local kappas have sharp increase in nuisance vars
#    - local kappa BFDR tends to be great around this time
# global kappas begin increasing; 
#    - seems like this would have been a good place to stop
#      (though still not really optimal or anything)

# need to look at global shrinkage param
# 
# ln_mode
# ln_fcn(
#   s_params$sa + s_params$sb, 
#   exp(s_params$sa_lvar) + exp(s_params$sb_lvar)
# )






