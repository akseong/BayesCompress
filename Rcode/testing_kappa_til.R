##################################################
## Project:   BFDR control via kappa
## Date:      Sep 30, 2025
## Author:    Arnie Seong
##################################################

# description: trying out BFDR controls via shrinkage factor kappa (local and usual)
# - apply to early simulations (linreg setting, overspecified linreg setting) as well as fcnl setting


#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))


# basic functions ---- 
BFDR <- function(dropout_probs, eta){
  delta_vec <- 1 * (dropout_probs <= eta)
  if (sum(delta_vec) == 0){ 
    warning("no included variables; returning BFDR = 0") 
    bfdr <- 0
  } else {
    bfdr <- sum((dropout_probs) * delta_vec) / sum(delta_vec)
  }
  return(
    list(
      "delta_i" = delta_vec,
      "bfdr" = bfdr
    )
  )
}

BFDR_eta_search <- function(dropout_probs, max_rate = 0.05){
  a_sort <- sort(dropout_probs)
  bfdrs <- sapply(a_sort, function(X) BFDR(dropout_probs, eta = X)$bfdr)
  inds <- which(bfdrs <= max_rate)
  if (length(inds)==0){
    warning("no threshold found, returning eta = 0")
    return(0)
  } else {
    a_sort[max(inds)]
  }
}

ln_mode <- function(mu, var){
  exp(mu - var)
}

ln_mean <- function(mu, var){
  exp(mu + var/2)
}

extract_kappa <- function(
    nn_model, 
    local_only = TRUE, 
    mode_or_mean = "mode", 
    want_zsq = FALSE
  ){
  # modes or means from variational posterior
  if (mode_or_mean == "mode"){
    ln_fcn <- ln_mode
  } else if (mode_or_mean == "mean"){
    ln_fcn <- ln_mean
  } else {stop("must choose variational posterior mode or mean")}
  
  #local shrinkage params
  at <- as_array(nn_model$fc1$atilde_mu)
  bt <- as_array(nn_model$fc1$btilde_mu)
  at_var <- exp(as_array(nn_model$fc1$atilde_logvar))
  bt_var <- exp(as_array(nn_model$fc1$btilde_logvar))
  
  if (local_only){
    # local shrinkage only
    zsq <- ln_fcn(at + bt, at_var + bt_var)
  } else {
    # local and global shrinkage
    sa <- as_array(nn_model$fc1$sa_mu)
    sb <- as_array(nn_model$fc1$sb_mu)
    sa_var <- exp(as_array(nn_model$fc1$sa_logvar))
    sb_var <- exp(as_array(nn_model$fc1$sb_logvar))
    zsq <- ln_fcn(at + bt + sa + sb, at_var + bt_var + sa_var + sb_var)
  }
  
  if (want_zsq){
    zsq
  } else {
    # return \kappa_i = 1/(1 + z^2)
    1/(1 + zsq)
  }
}

# basic function testing ---- 
nn_model <- torch::torch_load(here::here("sims", "results", "fcnl_hshoe_mod_12500obs_398060.pt"))

kappa <- extract_kappa(nn_model, local_only = FALSE)

eta <- BFDR_eta_search(kappa, max_rate = 0.01)
BFDR(kappa, eta)
sum(kappa <= eta)

eta <- BFDR_eta_search(kappa, max_rate = 0.05)
BFDR(kappa, eta)
sum(kappa <= eta)

eta <- BFDR_eta_search(kappa, max_rate = 0.1)
BFDR(kappa, eta)
sum(kappa <= eta)


#### using shrinkage factor Kappa based only on local shrinkage (i.e. tau = 1) ----
#### WORKS PRETTY DAMN WELL.  
kappa_til <- extract_kappa(nn_model, local_only = TRUE)

eta <- BFDR_eta_search(kappa_til, max_rate = 0.01)
BFDR(kappa_til, eta)
sum(kappa_til <= eta)

eta <- BFDR_eta_search(kappa_til, max_rate = 0.05)
BFDR(kappa_til, eta)
sum(kappa_til <= eta)

eta <- BFDR_eta_search(kappa_til, max_rate = 0.1)
BFDR(kappa_til, eta)
sum(kappa_til <= eta)



err_from_dropout <- function(
    dropout_vec, 
    max_bfdr = 0.01, 
    true_gam = c(rep(1, 4), rep(0, 100))
  ){
  eta <- BFDR_eta_search(dropout_vec, max_rate = max_bfdr)
  bfdr <- BFDR(dropout_vec, eta)$bfdr
  delta_i <- BFDR(dropout_vec, eta)$delta_i
  bin_err <- binary_err_rate(est = delta_i, tru = true_gam)
  fdr <- sum(delta_i - true_gam == 1) / sum(delta_i)
  c("fdr" = fdr, "bfdr" = bfdr, bin_err)
}




# comparison function ---- 
compare_kappas <- function(
    nn_model, 
    true_gam = c(rep(1, 4), rep(0, 100)),
    target_rates = c(0.1, 0.05, 0.01, 0.005, 0.001),
    use_weight_vars = TRUE,
    mode_or_mean = "mode"){
  
  kappa <- extract_kappa(nn_model, local_only = FALSE, mode_or_mean)
  kappa_til <- extract_kappa(nn_model, local_only = TRUE, mode_or_mean)
  
  if (use_weight_vars){
    w_lvars <- as_array(nn_model$fc1$weight_logvar)
    geom_means <- exp(colMeans(w_lvars))
    kappa <- kappa * geom_means
    kappa_til <- kappa_til * geom_means
  }
  
  res_global <- matrix(NA, nrow = length(target_rates), ncol = 7)
  colnames(res_global) <- c("nominal_rate", "calc_bfdr", "eta_thresh", "FP", "TP", "FN", "TN")
  res_local <- res_global
  
  # return for each nominal rate control: 
  # calculated bfdr, decision threshold, FP/TP/FN/TN
  # for both global and local kappa
  for (ind in 1:length(target_rates)){
    # global
    eta <- BFDR_eta_search(kappa, max_rate = target_rates[ind])
    bfdr <- BFDR(kappa, eta)$bfdr
    delta_i <- BFDR(kappa, eta)$delta_i
    err_rates <- binary_err_rate(est = delta_i, tru = true_gam)
    res_global[ind,] <- c(target_rates[ind], bfdr, eta, err_rates)
    
    # local
    eta <- BFDR_eta_search(kappa_til, max_rate = target_rates[ind])
    bfdr <- BFDR(kappa_til, eta)$bfdr
    delta_i <- BFDR(kappa_til, eta)$delta_i
    err_rates <- binary_err_rate(est = delta_i, tru = true_gam)
    res_local[ind,] <- c(target_rates[ind], bfdr, eta, err_rates)
  }
  
  return(
    list(
      "local" = res_local,
      "global" = res_global
    )
  )
}






#### examine linreg fits ----





#### examine fcnal fits ----

# not sure if these are models with fixed tau (good) or with tau as learnable param
# known to be good: 398060
# looks like models with seed 2494 is OK-ish?  maybe needs more training?

seeds <- c(445006, 79978, 2494, 398060)
seeds <- c(191578, 272393, 377047, 398060)

mod_stem <- here::here("sims", "results", "fcnl_hshoe_mod_12500obs_")
mod_fnames <- paste0(mod_stem, seeds, ".pt")
# checking known good mod
# load(paste0(mod_stem, "398060_continue.RData"))
# sim_res$alpha_mat[, 1:4]



global_res <- list()
local_res <- list()
for (mod_ind in 1:length(mod_fnames)){
  nn_model <- torch_load(mod_fnames[mod_ind])
  global_res[[mod_ind]] <- compare_kappas(nn_model)$global
  local_res[[mod_ind]] <- compare_kappas(nn_model)$local
}

global_res




##### using geom_means of weight variances
kappa <- extract_kappa(nn_model, local_only = FALSE)
w_lvars <- as_array(nn_model$fc1$weight_logvar)
gm <- exp(colMeans(w_lvars))
kappa_gm <- kappa * gm
kappa_til_gm <- kappa_til * gm

round(kappa, 3)
round(gm, 3)
round(kappa_gm, 3)
all.equal(kappa_gm, gm)
# basically the same.  (global kappas are mostly ~ 1) for nuisance vars




### Piironen + Vehtari 2017 "On the Hyperprior Choice for the Global Shrinkage Parameter in the Horseshoe Prior":
# posterior for kappa_k = 1 / (1 + n sigma^(-2) tau^2 lambda^2)
# in our simulations, test_mse ends up close to 1, n = 10k
# PERHAPS THIS EXPLAINS WHY tau gets so small but its OK.
# Basically we end up doing inference using the local dropout rate
post_kappa <- function(
    nn_model, 
    n_obs = 1e5,
    mse = 1,
    mode_or_mean = "mode", 
    want_zsq = FALSE
  ){
    # modes or means from variational posterior
    if (mode_or_mean == "mode"){
      ln_fcn <- ln_mode
    } else if (mode_or_mean == "mean"){
      ln_fcn <- ln_mean
    } else {stop("must choose variational posterior mode or mean")}
    
    #local shrinkage params
    at <- as_array(nn_model$fc1$atilde_mu)
    bt <- as_array(nn_model$fc1$btilde_mu)
    at_var <- exp(as_array(nn_model$fc1$atilde_logvar))
    bt_var <- exp(as_array(nn_model$fc1$btilde_logvar))
    
    # local and global shrinkage
    sa <- as_array(nn_model$fc1$sa_mu)
    sb <- as_array(nn_model$fc1$sb_mu)
    sa_var <- exp(as_array(nn_model$fc1$sa_logvar))
    sb_var <- exp(as_array(nn_model$fc1$sb_logvar))
    zsq <- ln_fcn(at + bt + sa + sb, at_var + bt_var + sa_var + sb_var)
    
    
    if (want_zsq){
      zsq
    } else {
      # return \kappa_i = 1/(1 + z^2)
      1/(1 + n_obs * (1/mse) * zsq)
    }
  }





pkappas <- post_kappa(nn_model)
round(pkappas, 3)

err_from_dropout(pkappas, max_bfdr = 0.001)
err_from_dropout(pkappas, max_bfdr = 0.005)
err_from_dropout(pkappas, max_bfdr = 0.01)
err_from_dropout(pkappas, max_bfdr = 0.02)
err_from_dropout(pkappas, max_bfdr = 0.05)
err_from_dropout(pkappas, max_bfdr = 0.1)



mod_fnames <- paste0(mod_stem, seeds, ".pt")
# checking known good mod
# load(paste0(mod_stem, "398060_continue.RData"))
# sim_res$alpha_mat[, 1:4]

max_rates <- c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
pkappas_res <- list()
kappa_gm_res <- list()

for (mod_ind in 1:length(mod_fnames)){
  nn_model <- torch_load(mod_fnames[mod_ind])
  pkappas <- post_kappa(nn_model)
  kappa <- extract_kappa(nn_model, local_only = FALSE, "mode")
  w_lvars <- as_array(nn_model$fc1$weight_logvar)
  gm <- exp(colMeans(w_lvars))
  kappa_gm <- kappa * gm
  
  pkap_err_mat <- t(sapply(max_rates, function(X) err_from_dropout(pkappas, max_bfdr = X)))
  kap_gm_err_mat <- t(sapply(max_rates, function(X) err_from_dropout(kappa_gm, max_bfdr = X)))
  rownames(pkap_err_mat) <- 
    rownames(kap_gm_err_mat) <- paste0("nom_rate: ", max_rates)
  pkappas_res[[mod_ind]] <- pkap_err_mat
  kappa_gm_res[[mod_ind]] <- kap_gm_err_mat
  
}

pkappas_res
kappa_gm_res

# similar results as using local kappa = 1 / (1 + lambda^2), i.e. local params only











pw_lvars <- as_array(nn_model$fc1$compute_posterior_param()$post_weight_var)
pw_gmvar <- exp(colMeans(pw_lvars))







# also want to extract global shrinkage params by model architecture





