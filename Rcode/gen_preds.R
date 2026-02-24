##################################################
## Project:   Sim analysis
## Date:      Dec 23, 2025
## Author:    Arnie Seong
##################################################

#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe_klcorrected.R"))
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))


## FCN INFERENCE ----

# load
stem_pvtau1 <- here::here("sims", "results", "hshoe_smooth_pvtau_1721632_12500obs_")
stem_pvtau2 <- here::here("sims", "results", "hshoe_smooth_pvtau_2721632_12500obs_")
seed_pvtau2 <- c(19709, 809872, 264744, 498729, 336130, 263808, 877164, 218489, 234821, 616240)
seed_pvtau1 <- c(561114, 453639, 663173, 108703, 165780)

resfile_pvtau1 <- paste0(stem_pvtau1, seed_pvtau1, ".RData")
modfile_pvtau1 <- paste0(stem_pvtau1, seed_pvtau1, ".pt")

resfile_pvtau2 <- paste0(stem_pvtau2, seed_pvtau2, ".RData")
modfile_pvtau2 <- paste0(stem_pvtau2, seed_pvtau2, ".pt")

res_fnames <- c(resfile_pvtau1, resfile_pvtau2)
mod_fnames <- c(modfile_pvtau1, modfile_pvtau2)
seeds <- c(seed_pvtau1, seed_pvtau2)

sim_ind <- 11
load(res_fnames[sim_ind])
nn_mod <- torch_load(mod_fnames[sim_ind])

# generate X for function list
sim_params <- sim_res$sim_params
xshow <- seq(-3, 3, length.out = 100)
curvmat <- matrix(0, ncol = length(sim_params$flist), nrow = length(sim_params$flist) * 100)
for (i in 1:length(sim_params$flist)){
  curvmat[1:length(xshow) + (i-1) * length(xshow), i] <- xshow
}
mat0 <- matrix(0, nrow = nrow(curvmat), ncol = sim_params$d_in - length(sim_params$flist))
x_plot <- torch_tensor(cbind(curvmat, mat0))


get_nn_mod_Ey <- function(nn_mod, X, ln_fcn = ln_mode, deterministic = FALSE){
  # pre_act
  num_layers <- length(nn_mod$children)
  input <- X
  for (nn_layer in 1:num_layers){

    ztilde <- ln_fcn(
      mu = as_array(nn_mod$children[[nn_layer]]$atilde_mu + nn_mod$children[[nn_layer]]$btilde_mu)/2,
      var = exp(as_array(nn_mod$children[[nn_layer]]$atilde_logvar + nn_mod$children[[nn_layer]]$btilde_logvar))/4
    )
    
    s <- ln_fcn(
      mu = as_array(nn_mod$children[[nn_layer]]$sa_mu + nn_mod$children[[nn_layer]]$sb_mu)/2,
      var = exp(as_array(nn_mod$children[[nn_layer]]$sa_logvar + nn_mod$children[[nn_layer]]$sb_logvar))/4
    )
    
    z <- torch_tensor(ztilde*s)
    
    Xz <- input*z
    
    mu_activations <- nnf_linear(
      input = ifelse(deterministic, input, Xz), 
      weight = nn_mod$children[[nn_layer]]$weight_mu, 
      bias = nn_mod$children[[nn_layer]]$bias_mu
    )
    
    if (nn_layer < num_layers){
      # input for next layer
      input <- nnf_relu(mu_activations)
    }
  }
  
  return(mu_activations)
}



# generate y_news and get sds, quantiles
generate_y_news <- function(nn_mod, X, n_reps = 100){
  drop(
    replicate(
      n = n_reps, 
      expr = as_array(nn_mod(X))
    )
  )
}

# 
# 
# 
# 
# X = x_plot
# qtiles <- c(0.025, .975)
# n_reps <- 100
# X_showcols <- 1:4

plot_pred_quantiles <- function(
    nn_mod, 
    X, 
    X_showcols = 1:4,
    qtiles = c(0.025, 0.975),
    n_reps = 100,
    ln_fcn = ln_mode
){
  # mean functions
  Ey <- as_array(
    get_nn_mod_Ey(nn_mod, X, ln_fcn)
  )
  
  # generate y_new predictions, sdevs, qtiles
  ynews <- generate_y_news(nn_mod, X, n_reps)
  sd_ynew <- apply(ynews, 1, sd)
  qtilemat <- t(apply(
    ynews, 1, 
    function(X) quantile(X, qtiles)
  ))
  
  # gather into df for plotting
  resmat <- cbind(
    Ey,
    sd_ynew,
    qtilemat,
    as_array(X[, X_showcols])
  )
  colnames(resmat) <- c(
    "Ey",
    "sd_ynew",
    paste0("p", qtiles),
    paste0("X", X_showcols)
  )
  df <- as_data_frame(resmat)
  
  Ey_plt <- df %>% 
    pivot_longer(
      cols = starts_with("X"),
      names_to = "fcn",
      values_to = "x"
    ) %>% 
    ggplot() + 
    geom_line(
      aes(
        y = Ey, 
        x = x,
        color = fcn
      )
    ) + 
    labs(title = "estimated mean functions")
  
  qtile_plt <- Ey_plt + 
    geom_ribbon(
      aes_string(
        x = "x",
        ymin = paste0("p", qtiles[1]),
        ymax = paste0("p", qtiles[2]),
        fill = "fcn"
      ),
      alpha = 0.2
    ) + 
    labs(
      subtitle = paste0(
        "with (", qtiles[1], ", ", qtiles[2], ") ",
        "prediction quantiles"
      )
    )
  
  return(
    list(
      "df" = df,
      "Ey_plt" = Ey_plt,
      "qtile_plt" = qtile_plt
    )
  )
}


qtile_plt <- plot_pred_quantiles(nn_mod, X = x_plot)$qtile_plt


og_fcn_plt <- plot_datagen_fcns(flist = sim_res$sim_params$flist)
qtile_plt + 
  geom_line(
    data = og_fcns$data,
    aes(
      y = value,
      x = x, 
      group = fcn
    ),
    color = "black",
    linetype = "dotted"
  )
og_fcn_plt
