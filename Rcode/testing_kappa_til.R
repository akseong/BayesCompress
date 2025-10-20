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



get_s_params <- function(nn_model_layer){
  sa <- as_array(nn_model_layer$sa_mu)
  sb <- as_array(nn_model_layer$sb_mu)
  sa_var <- exp(as_array(nn_model_layer$sa_logvar))
  sb_var <- exp(as_array(nn_model_layer$sb_logvar))
  return(
    list(
      "sa" = sa,
      "sb" = sb,
      "sa_var" = sa_var,
      "sb_var" = sb_var
      )
    )
}

get_ztil_params <- function(nn_model_layer){
  atil <- as_array(nn_model_layer$atilde_mu)
  btil <- as_array(nn_model_layer$btilde_mu)
  atil_var <- exp(as_array(nn_model_layer$atilde_logvar))
  btil_var <- exp(as_array(nn_model_layer$btilde_logvar))
  return(
    list(
      "at" = atil,
      "bt" = btil,
      "at_var" = atil_var,
      "bt_var" = btil_var
    )
  )
}

get_s_sq <- function(nn_model_layer, ln_fcn = ln_mode){
  s_params <- get_s(nn_model_layer)
  s_sq <- ln_fcn(s_params$sa + s_params$sb, s_params$sa_var + s_params$sb_var)
  return(s_sq)
}

get_ztil_sq <- function(nn_model_layer, ln_fcn = ln_mode){
  ztil_params <- get_ztil(nn_model_layer)
  ztil_sq <- ln_fcn(
    ztil_params$at + ztil_params$bt, 
    ztil_params$at_var + ztil_params$bt_var
  )
  return(ztil_sq)
}

get_s_sq(nn_model$fc3)
get_ztil_sq(nn_model$fc3)

get_kappas <- function(nn_model_layer, type = "global"){
  s_sq <- get_s_sq(nn_model_layer)
  ztil_sq <- get_ztil_sq(nn_model_layer)
  
  if (type == "global"){
    kappas <- 1 / ( 1 + s_sq*ztil_sq)
  } else if (type == "local"){
    kappas <- 1 / ( 1 + ztil_sq)
  } else {
    warning("type must be global or local")
  }
  
  return(kappas)
}

get_kappas(nn_model$fc2)

length(nn_model$children)

l <- paste0("fc", 1:(length(nn_model$children)))
nn_model[[eval(l[1])]]
nn_model$fc1$tau

### Piironen + Vehtari 2017 "On the Hyperprior Choice for the Global Shrinkage Parameter in the Horseshoe Prior":
# posterior for kappa_k = 1 / (1 + n sigma^(-2) tau^2 lambda^2)
# in our simulations, test_mse ends up close to 1, n = 10k
# PERHAPS THIS EXPLAINS WHY tau gets so small but its OK.
# Basically we end up doing inference using the local dropout rate
post_kappa <- function(
    nn_model, 
    n_obs = 1e5,
    mse = 1,
    ln_fcn = ln_mode, 
    use_weight_vars = FALSE,
    want_zsq = FALSE
  ){
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
    
    if (use_weight_vars){
      w_lvars <- as_array(nn_model$fc1$weight_logvar)
      gm <- exp(colMeans(w_lvars))
      zsq <- zsq * gm
    }
    
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
pkappas_gm_res <- list()
kappa_gm_res <- list()
alphas_res <- list()
marg_alphas_res <- list()
sqrt_alphas_res <- list()


for (mod_ind in 1:length(mod_fnames)){
  nn_model <- torch_load(mod_fnames[mod_ind])
  # posterior kappas
  pkappas <- post_kappa(nn_model, mse = 1, use_weight_vars = FALSE)
  pkappas_gm <- post_kappa(nn_model, mse = 1, use_weight_vars = TRUE)
  # global kappas weighted by geom mean of  variances of weights incident on covariate
  kappa <- extract_kappa(nn_model, local_only = FALSE, "mode")
  w_lvars <- as_array(nn_model$fc1$weight_logvar)
  gm <- exp(colMeans(w_lvars))
  kappa_gm <- kappa * gm
  # dropout factor alpha
  local_alphas <- as_array(nn_model$fc1$get_dropout_rates(type = "local"))
  marg_alphas <- as_array(nn_model$fc1$get_dropout_rates(type = "marginal"))
  # sqrt alpha
  sqrt_alphas <- sqrt(local_alphas)
  
  
  pkap_err_mat <- t(sapply(max_rates, function(X) err_from_dropout(pkappas, max_bfdr = X)))
  pkap_gm_err_mat <- t(sapply(max_rates, function(X) err_from_dropout(pkappas_gm, max_bfdr = X)))
  kap_gm_err_mat <- t(sapply(max_rates, function(X) err_from_dropout(kappa_gm, max_bfdr = X)))
  alphas_err_mat <- t(sapply(max_rates, function(X) err_from_dropout(local_alphas, max_bfdr = X)))
  marg_alphas_err_mat <- t(sapply(max_rates, function(X) err_from_dropout(marg_alphas, max_bfdr = X)))
  sqrt_alphas_err_mat <- t(sapply(max_rates, function(X) err_from_dropout(sqrt_alphas, max_bfdr = X)))
  
  rownames(pkap_err_mat) <- 
    rownames(pkap_gm_err_mat) <-
    rownames(kap_gm_err_mat) <- 
    rownames(alphas_err_mat) <-
    rownames(marg_alphas_err_mat) <-
    rownames(sqrt_alphas_err_mat) <- paste0("nom_rate: ", max_rates)
  pkappas_res[[mod_ind]] <- pkap_err_mat
  pkappas_gm_res[[mod_ind]] <- pkap_gm_err_mat
  kappa_gm_res[[mod_ind]] <- kap_gm_err_mat
  alphas_res[[mod_ind]] <- alphas_err_mat
  marg_alphas_res[[mod_ind]] <- marg_alphas_err_mat
  sqrt_alphas_res[[mod_ind]] <- sqrt_alphas_err_mat
}

pkappas_res
pkappas_gm_res
kappa_gm_res
alphas_res
marg_alphas_res
sqrt_alphas_res


# posterior kappas (pkappas) gives similar results as using local kappa = 1 / (1 + lambda^2), i.e. local params only
# multiplying kappa by gm works best.  WHY.
round(gm, 3)
round(kappa, 3)
# global kappas are mostly very close to 1

round(kappa_til, 2)
# local kappas vary much more


zsq <- extract_kappa(nn_model, local_only = FALSE, want_zsq = TRUE)

kz <- 1 / (1 + z)
round(kz, 3)

err_from_dropout(kz)

err_from_dropout(gm)

w_mu <- as_array(nn_model$fc1$weight_mu)
w_lvar <- as_array(nn_model$fc1$weight_logvar)
pw_mu <- as_array(nn_model$fc1$compute_posterior_param()$post_weight_mu)
pw_var <-as_array(nn_model$fc1$compute_posterior_param()$post_weight_var)
round(w_mu, 3)
round(exp(w_lvar), 3)
round(pw_mu, 3)
round(pw_var, 3) 

sa_mu_1 <- as_array(nn_model$fc1$sa_mu)
sb_mu_1 <- as_array(nn_model$fc1$sb_mu)
sa_lv_1 <- as_array(nn_model$fc1$sa_logvar)
sb_lv_1 <- as_array(nn_model$fc1$sb_logvar)

sa_mu_2 <- as_array(nn_model$fc2$sa_mu)
sb_mu_2 <- as_array(nn_model$fc2$sb_mu)
sa_lv_2 <- as_array(nn_model$fc2$sa_logvar)
sb_lv_2 <- as_array(nn_model$fc2$sb_logvar)

sa_mu_3 <- as_array(nn_model$fc3$sa_mu)
sb_mu_3 <- as_array(nn_model$fc3$sb_mu)
sa_lv_3 <- as_array(nn_model$fc3$sa_logvar)
sb_lv_3 <- as_array(nn_model$fc3$sb_logvar)


# variance of the activations is actually computed: (x * z_i) * weight_var
# z_i contributes the shrinkage.








pw_lvars <- as_array(nn_model$fc1$compute_posterior_param()$post_weight_var)
pw_gmvar <- exp(colMeans(pw_lvars))


gm


w_mus <- as_array(nn_model$fc1$weight_mu)
w_lvars <- as_array(nn_model$fc1$weight_logvar)
gm1 <- exp(colMeans(w_lvars))

w_lvars2 <- as_array(nn_model$fc2$weight_logvar)
gm2 <- exp(colMeans(w_lvars2))

w_lvars3 <- as_array(nn_model$fc3$weight_logvar)
gm3 <- exp(colMeans(w_lvars3))

round(gm1, 3)
round(gm2, 3)
round(gm3, 3)

s_sq <- get_s_sq(nn_model$fc1)
ztil_sq <- get_ztil_sq(nn_model$fc1)

z <- sqrt(s_sq) * sqrt(ztil_sq)
shrunk_mus <- t(apply(w_mus, 1, function(X) X*z))
round(shrunk_mus, 3)[1:5, ]

shrunk_vars <- t(apply(exp(w_lvars), 1, function(X) X*(z^2)))
round(shrunk_vars, 3)
vs <- apply(shrunk_vars, 2, max)
ks <- 1 / (1 + vs)
kz <- 1 / (1 + z^2)
kztil <- 1 / (1 + ztil_sq)
round(cbind(ks, kz, kztil), 3)
err_from_dropout(kztil)







# CHECK TRAINING HISTORY
seeds <- c(191578, 272393, 377047, 398060)

mod_stem <- here::here("sims", "results", "fcnl_hshoe_mod_12500obs_")
rdata_fnames <- paste0(mod_stem, seeds, ".RData")


load(rdata_fnames[3])

sim_res$loss_mat

s_sq_vec <- ln_mode(
  sim_res$sa_mu_vec + sim_res$sb_mu_vec, 
  sim_res$sa_logvar_vec + sim_res$sb_logvar_vec
)

ztil_sq_mat <- matrix(NA, nrow = nrow(sim_res$atilde_mu_mat), ncol = ncol(sim_res$atilde_mu_mat))
z_sq_mat <- ztil_sq_mat

for (i in 1:length(sim_res$sa_mu_vec)){
  ztil_sq_mat[i, ] <- ln_mode(
    sim_res$atilde_mu_mat[i, ] + sim_res$btilde_mu_mat[i, ], 
    sim_res$atilde_logvar_mat[i, ] + sim_res$btilde_logvar_mat[i, ]
  )
  z_sq_mat[i, ] <- ztil_sq_mat[i, ] * s_sq_vec[i]
}



kappa_local_mat <- 1 / (1 + ztil_sq_mat)
kappa_global_mat <- 1 / (1 + z_sq_mat)

local_errs <- t(apply(
  kappa_local_mat, 1, 
  function(X) err_from_dropout(dropout_vec = X, max_bfdr = 0.001)))
global_errs <- t(apply(
  function(X) err_from_dropout(dropout_vec = X, max_bfdr = 0.001)))



#### local alphas work well
alpha_errs_001 <- t(apply(
  sim_res$alpha_mat, 1,
  function(X) err_from_dropout(dropout_vec = X, max_bfdr = 0.001)))
alpha_errs_001
round(sim_res$alpha_mat[, 1:4], 3)
round(sim_res$alpha_mat[, 5:10], 3)

alpha_errs_01 <- t(apply(
  sim_res$alpha_mat, 1,
  function(X) err_from_dropout(dropout_vec = X, max_bfdr = 0.01)))

alpha_errs_05 <- t(apply(
  sim_res$alpha_mat, 1,
  function(X) err_from_dropout(dropout_vec = X, max_bfdr = 0.05)))

alpha_errs_001
alpha_errs_01
alpha_errs_05



#### marginal alphas
ztil_vars <- exp(sim_res$atilde_logvar_mat) + exp(sim_res$btilde_logvar_mat)
# sweep(matrix(0, nrow = 50, ncol = 104), 1,  + exp(sim_res$sa_logvar_vec))
z_vars <- sweep(ztil_lvars, 1, FUN="+", STATS = exp(sim_res$sa_logvar_vec) + exp(sim_res$sb_logvar_vec))
glob_alphas <- exp(z_vars / 4) - 1

glob_alpha_errs_001 <- t(apply(
  glob_alphas, 1,
  function(X) err_from_dropout(dropout_vec = X, max_bfdr = 0.001)))
glob_alpha_errs_01 <- t(apply(
  glob_alphas, 1,
  function(X) err_from_dropout(dropout_vec = X, max_bfdr = 0.01)))
glob_alpha_errs_05 <- t(apply(
  glob_alphas, 1,
  function(X) err_from_dropout(dropout_vec = X, max_bfdr = 0.05)))
glob_alpha_errs_001
glob_alpha_errs_01
glob_alpha_errs_05

1 / glob_alphas[45:50, 1:4]
1 / sim_res$alpha_mat[45:50, 1:4]


hist(1/(glob_alphas[50, 5:104]), breaks = 50)
hist(1/(sim_res$alpha_mat[50, 5:104]), breaks = 50)
ch <- rchisq(100000, df = 1)
hist(ch, breaks = 50)


chi_probs <- pchisq(1/glob_alphas[50, ], df = 16)
err_from_dropout((1-chi_probs), max_bfdr = 0.001)





##### use the LOG of the mode?
# known good model
nn_model <- torch::torch_load(paste0(mod_stem,"377047.pt"))
s_params <- get_s_params(nn_model$fc1)
ztil_params <- get_ztil_params(nn_model$fc1)

s_mu <- s_params$sa + s_params$sb
ztil_mu <- ztil_params$at + ztil_params$bt
z_mu <- s_mu + ztil_mu
s_mu
round(ztil_mu, 3)
round(z_mu, 3)

s_var <- s_params$sa_var + s_params$sb_var
ztil_var <- ztil_params$at_var + ztil_params$bt_var
z_var <- s_var + ztil_var
s_var
round(ztil_var, 3)
round(z_var, 3)
round(ln_mode(z_mu, z_var), 3)


#LUW 2017 use negative log-mode var_zi - mu_zi to prune:
z_mode <- ln_mode(z_mu, z_var)
round(z_mode, 3)
# put on standard normal scale?
# Probability of a standard normal var being closer to 0 than the mode of z
round(2*(pnorm(z_mode) - .5), 3)
round(2*(pnorm(z_mode, sd = sqrt(1.3e-5)) - .5), 3)

pips <- 2*(pnorm(z_mode, sd = sqrt(1e-5)) - .5)

err_from_dropout(1-pips, max_bfdr = 0.001)
err_from_dropout(1-pips, max_bfdr = 0.005)
err_from_dropout(1-pips, max_bfdr = 0.01)
err_from_dropout(1-pips, max_bfdr = 0.05)
err_from_dropout(1-pips, max_bfdr = 0.1)
err_from_dropout(1-pips, max_bfdr = 0.15)
err_from_dropout(1-pips, max_bfdr = 0.2)
err_from_dropout(1-pips, max_bfdr = 0.25)
err_from_dropout(1-pips, max_bfdr = 0.3)
err_from_dropout(1-pips, max_bfdr = 0.4)
err_from_dropout(1-pips, max_bfdr = 0.5)
err_from_dropout(1-pips, max_bfdr = 0.6)
err_from_dropout(1-pips, max_bfdr = 0.7)
err_from_dropout(1-pips, max_bfdr = 0.8)
err_from_dropout(1-pips, max_bfdr = 0.9)
#### WORKS WELL, BFDR MATCHES FDR well (for sparse model)


#### works for known good model that has finished running.  want to see interim training results.
ids <- c(966694, 191578, 272393, 718069, 377047)
load(paste0(mod_stem, ids[2], ".RData"))
test_mse_vec <- sim_res$loss_mat[, 3]
test_mse_vec
# sim 1 test mse: 3.21  XX
# sim 2: 1.2
# sim 3: 3.3    XX
# sim 4: 1.05
# sim 5: 1.07
# should be close to 1

sim_res$sim_params$sim_seeds
s_mu_vec <- sim_res$sa_mu_vec + sim_res$sb_mu_vec
s_var_vec <- exp(sim_res$sa_logvar_vec) + exp(sim_res$sb_logvar_vec)
ztil_mu_mat <- sim_res$atilde_mu_mat + sim_res$btilde_mu_mat
ztil_var_mat <- exp(sim_res$atilde_logvar_mat) + exp(sim_res$btilde_logvar_mat)
z_var_mat <- sweep(ztil_var_mat, 1, FUN="+", STATS = s_var_vec)
# test sweep:
# test <- matrix(0, nrow = nrow(ztil_var_mat), ncol = nrow(ztil_var_mat))
# sweep(test, 1, FUN="+", STATS = 1:nrow(ztil_var_mat))
z_mu_mat <- sweep(ztil_mu_mat, 1, FUN="+", STATS = s_mu_vec)

# z_mode_mat <- ln_mode(z_mu_mat/2, z_var_mat/4) # the mode of z
z_mode_mat <- ln_mode(z_mu_mat, z_var_mat)  # this is actually the mode of z^2

pips_mat <- matrix(NA, nrow = nrow(ztil_var_mat), ncol = ncol(ztil_var_mat))
err_mat <- matrix(NA, nrow = nrow(ztil_var_mat), ncol = 6)
colnames(err_mat) <- names(err_from_dropout(1-pips, max_bfdr = 0.9))


bfdrs <- c(.001, .01, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5)
err_array <- array(NA, dim = c(50, 6, length(bfdrs)))


for (j in 1:length(bfdrs)){
  for (i in 1:nrow(z_var_mat)){
    pips_i <- 2*(pnorm(z_mode_mat[i, ], sd = sqrt(test_mse_vec[i]*1e-5)) - .5)  # use with zsq
    # pips_i <- 2*(pnorm(z_mode_mat[i, ], sd = sqrt(test_mse_vec[i])) - .5)       # use with z
    # pips_i <- pchisq(z_mode_mat[i, ], df = 1)   # use with zsq; extremely conservative
    pips_mat[i, ] <- pips_i
    err_mat[i, ] <- err_from_dropout(1-pips_i, max_bfdr = bfdrs[j])
  } 
  err_array[, , j] <- err_mat
}

# across epochs, bfdr = .001
err_array[, , 1]

# across epochs, bfdr = .05
err_array[, , 3]

# across bfdrs
bfdrs
t(err_array[50, ,])



#### Pretty stable!



# check test_mses across runs:

ids <- c(966694, 191578, 272393, 718069, 377047)
test_mse_mat <- 
  train_mse_mat <- 
  kl_mat <- matrix(NA, ncol = length(ids), nrow = 50)


for (i in 1:length(ids)){
  load(paste0(mod_stem, ids[i], ".RData"))
  kl_mat[, i] <- sim_res$loss_mat[, 1]
  train_mse_mat[, i] <- sim_res$loss_mat[, 2]
  test_mse_mat[, i] <- sim_res$loss_mat[, 3]
}

round(kl_mat, 3)
round(train_mse_mat, 3)
round(test_mse_mat, 3)






########################## mode of z^2 the same as (mode z) ^2

#### mode of z
nn_model <- torch::torch_load(paste0(mod_stem,"377047.pt"))
s_params <- get_s_params(nn_model$fc1)
ztil_params <- get_ztil_params(nn_model$fc1)

s_mu <- 1/2 * (s_params$sa + s_params$sb)
ztil_mu <- 1/2 * (ztil_params$at + ztil_params$bt)
z_mu <- s_mu + ztil_mu
s_mu
round(ztil_mu, 3)
round(z_mu, 3)

s_var <- 1/4 * (s_params$sa_var + s_params$sb_var)
ztil_var <- 1/4 * (ztil_params$at_var + ztil_params$bt_var)
z_var <- s_var + ztil_var
s_var
round(ztil_var, 3)
round(z_var, 3)
round(ln_mode(z_mu, z_var), 3)


#LUW 2017 use negative log-mode var_zi - mu_zi to prune:
z_mode <- ln_mode(z_mu, z_var)




#### mode of z^2
nn_model <- torch::torch_load(paste0(mod_stem,"377047.pt"))
s_params <- get_s_params(nn_model$fc1)
ztil_params <- get_ztil_params(nn_model$fc1)

ssq_mu <- s_params$sa + s_params$sb
ztilsq_mu <- ztil_params$at + ztil_params$bt
zsq_mu <- ssq_mu + ztilsq_mu
ssq_mu
round(ztilsq_mu, 3)
round(zsq_mu, 3)

ssq_var <- s_params$sa_var + s_params$sb_var
ztilsq_var <- ztil_params$at_var + ztil_params$bt_var
zsq_var <- ssq_var + ztilsq_var
ssq_var
round(ztilsq_var, 3)
round(zsq_var, 3)
round(ln_mode(zsq_mu, zsq_var), 3)


#LUW 2017 use negative log-mode var_zi - mu_zi to prune:
zsq_mode <- ln_mode(zsq_mu, zsq_var)

z_mode
zsq_mode

round(cbind(z_mode^2, zsq_mode), 3)

pips_kappa <- 1 - 1 / (1 + z_mode^2)

z_modesq <- z_mode^2


## these are for w-TILDE, not w (w = wtil * z)
wvars <- exp(as_array(nn_model$fc1$weight_logvar))
gm_wvars <- exp(colMeans(as_array(nn_model$fc1$weight_logvar)))
absmean_wmus <- colMeans(abs(as_array(nn_model$fc1$weight_mu)))

ztil_sq_layer2 <- get_ztil_sq(nn_model$fc2)
s_sq_layer2 <- get_s_sq(nn_model$fc2)
zsq_2 <- ztil_sq_layer2 * s_sq_layer2
kappas_layer2 <- 1 / (1 + round(zsq_2, 3))
kappas_layer2

round(wvars, 3)[1:5, ]
round(z_modesq, 3)
round(pips_kappa, 3)

kappas_layer1 <- 1 / (1 + z_mode^2)
kappas_layer1
kappas_layer2


# a sane variable inclusion measure would need to consider
# the mean of the weights as well as the variance.
# 
round(cbind(absmean_wmus, gm_wvars, z_modesq), 3)
round(cbind(absmean_wmus * z_mode, gm_wvars * z_modesq), 3)
# does it make more sense to construct a test based on the WEIGHT mean and sd?

W_mus <- absmean_wmus * z_mode
W_vars <- gm_wvars * z_mode^2

wald <- W_mus^2 / W_vars
pips_w <- pchisq(wald, df = 1)
err_from_dropout(1 - pips_w, max_bfdr = 0.01)
err_from_dropout(1 - pips_w, max_bfdr = 0.05)
err_from_dropout(1 - pips_w, max_bfdr = 0.1)
err_from_dropout(1 - pips_w, max_bfdr = 0.15)
err_from_dropout(1 - pips_w, max_bfdr = 0.2)
err_from_dropout(1 - pips_w, max_bfdr = 0.25)
err_from_dropout(1 - pips_w, max_bfdr = 0.3)

##### NOT BAD, and at least it's defensible.
### can it get better using only those neurons in the first layer that weren't eliminated in the 2nd layer?
### (wouldn't affect the means as much?  but might enlarge nuisance var means?)

