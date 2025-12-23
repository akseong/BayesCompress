##################################################
## Project:   Sim analysis
## Date:      Dec 23, 2025
## Author:    Arnie Seong
##################################################

# Kaiming initialization used
# normalize train data

#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe_klcorrected.R"))
source(here("Rcode", "sim_functions.R"))
# source(here("Rcode", "sim_hshoe_normedresponse.R"))

if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}



fname_stem <- here::here("sims", "results", "hshoe_smooth_pvtau_1721632_12500obs_")
load(paste0(fname_stem, "108703.RData"))

sim_res$loss_mat
sim_res$mod_path
sim_res$sim_params$sim_seeds


# PLOTTING over epochs ----

## loss_pltfcn
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

# ### usage 
# loss_pltfcn(sim_res, burn = 0)$mse_plt
# loss_pltfcn(sim_res, burn = 8e5)$mse_plt
# 
# loss_pltfcn(sim_res, burn = 0)$kl_plt  + 
#   theme(legend.position = "none")
# loss_pltfcn(sim_res, burn = 5e5)$kl_plt  + 
#   theme(legend.position = "none")



## varmat_pltfcn 
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

### usage
varmat_pltfcn(
  sim_res$alpha_mat,
  y_name = "alpha",
  burn = 0
)$all_vars_plt

varmat_pltfcn(
  sim_res$alpha_mat,
  y_name = "alpha",
  burn = 25e4
)$show_vars_plt


varmat_pltfcn(
  sim_res$kappa_mat,
  y_name = "GLOBAL kappa",
  burn = 0
)$all_vars_plt

varmat_pltfcn(
  sim_res$kappa_local_mat,
  y_name = "LOCAL kappa",
  burn = 0
)$all_vars_plt

varmat_pltfcn(
  sim_res$kappa_local_mat,
  y_name = "LOCAL kappa",
  burn = 25e4
)$show_vars_plt



# DECISIONS over epoch ---- 

dropout_mat <- sim_res$kappa_local_mat

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

err_by_epoch(
  dropout_mat = sim_res$kappa_local_mat,
  max_bfdr = 0.01
)

err_by_epoch(
  dropout_mat = sim_res$alpha_mat,
  max_bfdr = 0.01
)

err_by_epoch(
  dropout_mat = sim_res$kappa_mat,
  max_bfdr = 0.01
)

err_by_epoch(
  dropout_mat = sim_res$kappa_local_mat,
  max_bfdr = 0.01
)


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

ln_mode
ln_fcn(
  s_params$sa + s_params$sb, 
  exp(s_params$sa_lvar) + exp(s_params$sb_lvar)
)


