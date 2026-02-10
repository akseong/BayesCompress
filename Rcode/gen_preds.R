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


get_nn_mod_Ey <- function(nn_mod, X, ln_fcn = ln_mode){
  
  
  # pre_act
  num_layers <- length(nn_mod$children)
  input <- X
  for (nn_layer in 1:num_layers){
  
    # log_atilde <- reparameterize(mu = nn_mod$children[[nn_layer]]$atilde_mu, logvar = nn_mod$children[[nn_layer]]$atilde_logvar)
    # log_btilde <- reparameterize(mu = nn_mod$children[[nn_layer]]$btilde_mu, logvar = nn_mod$children[[nn_layer]]$btilde_logvar)
    # log_sa <- reparameterize(mu = nn_mod$children[[nn_layer]]$sa_mu, logvar = nn_mod$children[[nn_layer]]$sa_logvar)
    # log_sb <- reparameterize(mu = nn_mod$children[[nn_layer]]$sb_mu, logvar = nn_mod$children[[nn_layer]]$sb_logvar)
    # log_s <- 1/2 * (log_sa + log_sb)
    # log_ztilde <- 1/2 * (log_atilde + log_btilde)
    # 
    # log_s$exp() * log_ztilde$exp()

    ztilde <- ln_fcn(mu = as_array(nn_mod$children[[nn_layer]]$atilde_mu + nn_mod$children[[nn_layer]]$btilde_mu)/2,
            var = exp(as_array(nn_mod$children[[nn_layer]]$atilde_logvar + nn_mod$children[[nn_layer]]$btilde_logvar))/4
    )
    
    s <- ln_fcn(mu = as_array(nn_mod$children[[nn_layer]]$sa_mu + nn_mod$children[[nn_layer]]$sb_mu)/2,
            var = exp(as_array(nn_mod$children[[nn_layer]]$sa_logvar + nn_mod$children[[nn_layer]]$sb_logvar))/4
    )
    
    # z<- (log_s + log_ztilde)$exp()
    
    z <- torch_tensor(ztilde*s)
    
    Xz <- input*z
    
    mu_activations <- nnf_linear(
      input = Xz, 
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


get_nn_mod_y_new <- function(nn_mod, X, sampling = TRUE){
  
  # pre_act
  num_layers <- length(nn_mod$children)
  input <- X
  for (nn_layer in 1:num_layers){
    
    log_atilde <- reparameterize(mu = nn_mod$children[[nn_layer]]$atilde_mu, logvar = nn_mod$children[[nn_layer]]$atilde_logvar)
    log_btilde <- reparameterize(mu = nn_mod$children[[nn_layer]]$btilde_mu, logvar = nn_mod$children[[nn_layer]]$btilde_logvar)
    log_sa <- reparameterize(mu = nn_mod$children[[nn_layer]]$sa_mu, logvar = nn_mod$children[[nn_layer]]$sa_logvar)
    log_sb <- reparameterize(mu = nn_mod$children[[nn_layer]]$sb_mu, logvar = nn_mod$children[[nn_layer]]$sb_logvar)
    log_s <- 1/2 * (log_sa + log_sb)
    log_ztilde <- 1/2 * (log_atilde + log_btilde)
    log_s$exp() * log_ztilde$exp()
    
    z <- (log_s + log_ztilde)$exp()
    Xz <- input*z
    
    mu_activations <- nnf_linear(
      input = Xz, 
      weight = nn_mod$children[[nn_layer]]$weight_mu, 
      bias = nn_mod$children[[nn_layer]]$bias_mu
    )
    
    var_activations <- nnf_linear(
      input = xz$pow(2), 
      weight = nn_mod$children[[nn_layer]]$weight_logvar$exp(), 
      bias = nn_mod$children[[nn_layer]]$bias_logvar$exp()
    )
    
    activations <- reparameterize(
        mu = mu_activations, 
        logvar = var_activations$log(), 
        sampling = sampling
      )
    
    if (nn_layer < num_layers){
      # input for next layer
      input <- nnf_relu(activations)
    }
  }
  
  return(activations)
}  



ynew <- get_nn_mod_y_new(nn_mod, x_plot, sampling = TRUE)

yhat <- get_nn_mod_Ey(nn_mod, X = x_plot, ln_fcn = ln_mean)
df <- cbind(as_array(yhat), as_array(x_plot))
colnames(df) <- c("yhat", paste0("x", 1:ncol(as_array(x_plot))))
as_data_frame(df) %>% 
    select(1:5) %>% 
    pivot_longer(cols = -yhat) %>% 
    ggplot(aes(y = yhat, x = value, color = name)) + 
    geom_line() + 
  labs(
    title = "yhat functions"
  )

  
  
  # # deterministic version in original code
  # nnf_linear(
  #   input = X, 
  #   weight = nn_mod$children[[nn_layer]]$weight_mu, 
  #   bias = nn_mod$children[[nn_layer]]$bias_mu
  # )
  
  
  

