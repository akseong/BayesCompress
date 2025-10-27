##################################################
## Project:   retrain reduced mod
## Date:      Oct 22, 2025
## Author:    Arnie Seong
##################################################

# setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe_cudafix.R"))
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

FDR <- function(delta_vec, true_gam = c(rep(1, 4), rep(0, 100))){
  if (sum(delta_i) == 0){
    return(0)
  } else {
    return(sum((delta_i - true_gam) == 1) / sum(delta_i))
  }
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

err_from_dropout <- function(
    dropout_vec, 
    max_bfdr = 0.01, 
    true_gam = c(rep(1, 4), rep(0, 100))
){
  eta <- BFDR_eta_search(dropout_vec, max_rate = max_bfdr)
  bfdr <- BFDR(dropout_vec, eta)$bfdr
  delta_i <- BFDR(dropout_vec, eta)$delta_i
  bin_err <- binary_err_rate(est = delta_i, tru = true_gam)
  fdr <- FDR(delta_vec = delta_i, true_gam = true_gam)
  c("fdr" = fdr, "bfdr" = bfdr, bin_err)
}

get_s_params <- function(nn_model_layer){
  sa <- as_array(nn_model_layer$sa_mu)
  sb <- as_array(nn_model_layer$sb_mu)
  sa_lvar <- as_array(nn_model_layer$sa_logvar)
  sb_lvar <- as_array(nn_model_layer$sb_logvar)
  return(
    list(
      "sa" = sa,
      "sb" = sb,
      "sa_lvar" = sa_lvar,
      "sb_lvar" = sb_lvar
    )
  )
}

get_ztil_params <- function(nn_model_layer){
  atil <- as_array(nn_model_layer$atilde_mu)
  btil <- as_array(nn_model_layer$btilde_mu)
  atil_lvar <- as_array(nn_model_layer$atilde_logvar)
  btil_lvar <- as_array(nn_model_layer$btilde_logvar)
  return(
    list(
      "at" = atil,
      "bt" = btil,
      "at_lvar" = atil_lvar,
      "bt_lvar" = btil_lvar
    )
  )
}

get_s_sq <- function(nn_model_layer, ln_fcn = ln_mode){
  s_params <- get_s_params(nn_model_layer)
  s_sq <- ln_fcn(
    s_params$sa + s_params$sb, 
    exp(s_params$sa_lvar) + exp(s_params$sb_lvar)
  )
  return(s_sq)
}

get_ztil_sq <- function(nn_model_layer, ln_fcn = ln_mode){
  ztil_params <- get_ztil_params(nn_model_layer)
  ztil_sq <- ln_fcn(
    ztil_params$at + ztil_params$bt, 
    exp(ztil_params$at_lvar) + exp(ztil_params$bt_lvar)
  )
  return(ztil_sq)
}

get_kappas <- function(nn_model_layer, type = "global"){
  
  ztil_sq <- get_ztil_sq(nn_model_layer)
  
  if (type == "global"){
    s_sq <- get_s_sq(nn_model_layer)
    kappas <- 1 / ( 1 + s_sq*ztil_sq)
  } else if (type == "local"){
    kappas <- 1 / ( 1 + ztil_sq)
  } else {
    warning("type must be global or local")
  }
  
  return(kappas)
}

get_wtil_params <- function(nn_model_layer){
  wtil_lvar <- as_array(nn_model_layer$weight_logvar)
  wtil_mu <- as_array(nn_model_layer$weight_mu)
  
  return(
    list(
      "wtil_lvar" = wtil_lvar,
      "wtil_mu" = wtil_mu
    )
  )
}


get_Wz_params <- function(nn_model_layer){
  # these are the params for the CONDITIONAL W | z 
  wtil_params <- get_wtil_params(nn_model_layer)
  z_sq <- get_s_sq(nn_model_layer) * get_ztil_sq(nn_model_layer)
  # # checking sweep function
  # # want to multiply test_mat column j by element j in mult_vec
  # test_mat <- cbind(
  #   c(0,0,0,0,0),
  #   c(1, 1, 1, 1, 1),
  #   c(-1, -1, -1, -1, -1)
  # )
  # test_mat
  # mult_vec <- 1:3
  # sweep(test_mat, 2, STATS = mult_vec, FUN = "*")
  Wz_mu <- sweep(
    wtil_params$wtil_mu, 
    MARGIN = 2, 
    STATS = sqrt(z_sq), 
    FUN = "*"
  )
  Wz_var <- sweep(
    exp(wtil_params$wtil_lvar), 
    MARGIN = 2, 
    STATS = z_sq, 
    FUN = "*"
  )
  
  return(
    list(
      "Wz_mu" = Wz_mu,
      "Wz_var" = Wz_var
    )
  )
}



# display params:
seeds <- c(966694, 191578, 272393, 718069, 377047)
mod_stem <- here::here("sims", "results", "fcnl_hshoe_mod_12500obs_")
mod_fnames <- paste0(mod_stem, seeds, ".pt")
res_fnames <- paste0(mod_stem, seeds, ".RData")

k1_mat <- matrix(NA, nrow = length(seeds), ncol = 104)
k2_mat <- matrix(NA, nrow = length(seeds), ncol = 16)
k3_mat <- matrix(NA, nrow = length(seeds), ncol = 8)
s_sq_mat <- matrix(NA, nrow = length(seeds), ncol = 3)
ztil_sq1_mat <- matrix(NA, nrow = length(seeds), ncol = 104)
ztil_sq2_mat <- matrix(NA, nrow = length(seeds), ncol = 16)
ztil_sq3_mat <- matrix(NA, nrow = length(seeds), ncol = 8)
final_loss <- matrix(NA, nrow = length(seeds), ncol = 3)
rnames <- paste0("mod_", seeds)
rownames(k1_mat) <- rownames(k2_mat) <- rownames(k3_mat) <- 
  rownames(ztil_sq1_mat) <- rownames(ztil_sq2_mat) <- rownames(ztil_sq3_mat) <- 
  rownames(s_sq_mat) <- rownames(final_loss) <- rnames
colnames(s_sq_mat) <- paste0("layer_", 1:3)
colnames(final_loss) <- c("kl", "mse", "test_mse")

for (i in 1:length(seeds)){
  nn_model <- torch_load(mod_fnames[i])
  load(res_fnames[i])
  
  final_loss[i, ] <- sim_res$loss_mat[nrow(sim_res$loss_mat), ]
  
  s_sq_mat[i, ] <- c(
    get_s_sq(nn_model$fc1),
    get_s_sq(nn_model$fc2),
    get_s_sq(nn_model$fc3)
  )
  
  ztil_sq1_mat[i, ] <- get_ztil_sq(nn_model$fc1)
  ztil_sq2_mat[i, ] <- get_ztil_sq(nn_model$fc2)
  ztil_sq3_mat[i, ] <- get_ztil_sq(nn_model$fc3)
  
  k1_mat[i, ] <- get_kappas(nn_model$fc1)
  k2_mat[i, ] <- get_kappas(nn_model$fc2)
  k3_mat[i, ] <- get_kappas(nn_model$fc3)
}

round(final_loss, 3)  # at least 2 have not converged (1, 3)
round(s_sq_mat, 5)
round(ztil_sq1_mat, 3)
round(ztil_sq2_mat, 3)
round(ztil_sq3_mat, 3)



round(k1_mat, 3)
round(k2_mat, 1)
round(k3_mat, 2)
rowSums(k1_mat < .9)
rowSums(k2_mat < .9)
rowSums(k3_mat < .9)


k1means <- rowMeans(k1_mat)
k2means <- rowMeans(k2_mat)
k3means <- rowMeans(k3_mat)


k1_sum <- k2_sum <- k3_sum <- rep(NA, 4)
for (i in 1:4){
  k1_sum[i] <- sum(k1_mat[i,] < k1means[i])
  k2_sum[i] <- sum(k2_mat[i,] < k2means[i])
  k3_sum[i] <- sum(k3_mat[i,] < k3means[i])
}

k1_sum
k2_sum
k3_sum


# model 377047 has high k2's, i.e. only 1 is below .9.  Check its predictions.
seeds
nn_model <- torch::torch_load(mod_fnames[5], device = "cpu")

load(res_fnames[5])
sim_res$loss_mat

set.seed(seeds[5])
torch_manual_seed(seeds[5])
simdat <- sim_func_data(
  n_obs = sim_res$sim_params$n_obs, 
  d_in = sim_res$sim_params$d_in, 
  flist = sim_res$sim_params$flist, 
  err_sigma = sim_res$sim_params$err_sig,
)
if (nn_model$fc1$atilde_logvar$is_cuda){
  simdat$x <- simdat$x$to(device = "cuda")
  simdat$y <- simdat$x$to(device = "cuda")
}

pred_mats <- make_pred_mats(
  flist = sim_res$sim_params$flist,
  d_in = sim_res$sim_params$d_in
)

fcn_plt <- plot_datagen_fcns(
  flist = sim_res$sim_params$flist,
  min_x = -5,
  max_x = 5,
  x_length = 500
)

plot_fcn_preds(torchmod = nn_model, pred_mats, want_df = FALSE, want_plot = TRUE)


#### Why are estimates for functions 1-3 so much higher?----
# add noise to nuisance vars since
# original data has normal noise in covariates 5-104
pred_mats_noisy <- pred_mats
pred_mats_noisy$x_tensor[, 5:104] <- torch_normal(
  mean = 0, std = 1, 
  size = c(nrow(pred_mats$x_tensor), 100))
plot_fcn_preds(torchmod = nn_model, pred_mats_noisy, want_df = FALSE, want_plot = TRUE)
# bias issue persists

# 0 out entire x matrix
pred_mats_0s <- pred_mats
pred_mats_0s$x_tensor <- torch_zeros(size = c(nrow(pred_mats$x_tensor), ncol(pred_mats$x_tensor)))
plot_fcn_preds(torchmod = nn_model, pred_mats_0s, want_df = FALSE, want_plot = TRUE)
# there is a global bias upwards???? 

# 0 x1-x4, noisy elsewhere
pred_mats_0cov <- pred_mats
pred_mats_0cov$x_tensor[, 5:104] <- torch_normal(
  mean = 0, std = 1, 
  size = c(nrow(pred_mats$x_tensor), 100))

pred_mats_0cov$x_tensor[, 1:4] <- torch_zeros(size = c(nrow(pred_mats$x_tensor), 4))
plot_fcn_preds(torchmod = nn_model, pred_mats_0cov, want_df = FALSE, want_plot = TRUE)

# check 0ing out other covs from original data
y <- simdat$y
x1_ord <- as_array(simdat$x[, 1]$sort()[[2]])
x2_ord <- as_array(simdat$x[, 2]$sort()[[2]])
x3_ord <- as_array(simdat$x[, 3]$sort()[[2]])
x4_ord <- as_array(simdat$x[, 4]$sort()[[2]])

pred_x1 <- simdat$x[x1_ord, ]
pred_x2 <- simdat$x[x2_ord, ]
pred_x3 <- simdat$x[x3_ord, ]
pred_x4 <- simdat$x[x4_ord, ]

pred_x1[, 2:104] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x2[, 1] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x2[, 3:104] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x3[, 1:2] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x3[, 4:104] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x4[, 1:3] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x4[, 5:104] <- torch_zeros(size = c(nrow(pred_x1), 1))

yhat1 <- as_array(nn_model(pred_x1))
plot(yhat1 ~ as_array(pred_x1[, 1]))

yhat2 <- as_array(nn_model(pred_x2))
plot(yhat2 ~ as_array(pred_x2[, 2]))

yhat3 <- as_array(nn_model(pred_x3))
plot(yhat3 ~ as_array(pred_x3[, 3]))

yhat4 <- as_array(nn_model(pred_x4))
plot(yhat4 ~ as_array(pred_x4[, 4]))


# try keeeping covariate 4 in for all of them


pred_x1 <- simdat$x[x1_ord, ]
pred_x2 <- simdat$x[x2_ord, ]
pred_x3 <- simdat$x[x3_ord, ]
pred_x4 <- simdat$x[x4_ord, ]

pred_x1[, 2:3] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x1[, 5:104] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x2[, 1] <- torch_zeros(size = nrow(pred_x1))
pred_x2[, 3] <- torch_zeros(size = nrow(pred_x1))
pred_x2[, 5:104] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x3[, 1:2] <- torch_zeros(size = c(nrow(pred_x1), 1))
pred_x3[, 5:104] <- torch_zeros(size = c(nrow(pred_x1), 1))

yhat1 <- as_array(nn_model(pred_x1))
plot(yhat1 ~ as_array(pred_x1[, 1]))

yhat2 <- as_array(nn_model(pred_x2))
plot(yhat2 ~ as_array(pred_x2[, 2]))

yhat3 <- as_array(nn_model(pred_x3))
plot(yhat3 ~ as_array(pred_x3[, 3]))

yhat4 <- as_array(nn_model(pred_x4))
plot(yhat4 ~ as_array(pred_x4[, 4]))

plot_datagen_fcns(sim_res$sim_params$flist)

# sharpness of function 4 is probably causing problems, especially
# because x-values are centered around 0.  
# Replace with long-period sine / cos wave?  Or polynomial function?





# load known good model ----




k1 <- get_kappas(nn_model$fc1)
round(k1, 3)

# selecting nodes with dropout probability < 0.5 (median model)
k2 <- get_kappas(nn_model$fc2)
round(k2, 3)
l1_selected_nodes <- k2 < 0.95

k3 <- get_kappas(nn_model$fc3)
round(k3, 3)  
l2_selected_nodes <- k3 < 0.95


red_mod <- nn_model

red_mod$fc1$weight_logvar <- nn_model$fc1$weight_logvar[l1_selected_nodes, ]
red_mod$fc1$weight_mu <- nn_model$fc1$weight_mu[l1_selected_nodes]

red_mod$fc2$weight_logvar <- nn_model$fc2$weight_logvar[l2_selected_nodes, ]
red_mod$fc2$weight_mu <- nn_model$fc2$weight_mu[l2_selected_nodes]

nn_model$children
red_mod$children














# visualization ----
make_pred_mats

plot_datagen_fcns

plot_fcn_preds









# comparisons
get_lm_stats






