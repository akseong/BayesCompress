##################################################
## Project:   intercept investigation
## Date:      Nov 03, 2025
## Author:    Arnie Seong
##################################################


library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))


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



round(k2_mat, 2)

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
seednum <- 2
nn_model <- torch::torch_load(mod_fnames[seednum], device = "cpu")

load(res_fnames[seednum])
sim_res$loss_mat

set.seed(seeds[seednum])
torch_manual_seed(seeds[seednum])
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
pred_x2[, 1] <- torch_zeros(size = nrow(pred_x1))
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
# these are all fairly crisp lines


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

plot_datagen_fcns(sim_res$sim_params$flist)
# sharpness of function 4 is probably causing problems, especially
# because x-values are centered around 0 (sharp peak).  
# Replace with long-period sine / cos wave?  Or polynomial function?






# NON-REDUCED, known good model ----
seeds
seednum <- 2 # 191578 seems OK
nn_model <- torch::torch_load(mod_fnames[seednum], device = "cpu")

s_params <- get_s_params(nn_model$fc1)
# p(s_a) = gamma, q(s_a) = LN
# s_b: IG, LN
# atilde: gamma, LN
# btilde: IG, LN
ln_mean(mu = s_params$sa, var = exp(s_params$sa_lvar))
lsa_samp <- rnorm(1000, mean = s_params$sa, sd = exp(s_params$sa_lvar/2))
hist(exp(lsa_samp))

ln_mean(mu = s_params$sb, var = exp(s_params$sb_lvar))
lsb_samp <- rnorm(1000, mean = s_params$sb, sd = exp(s_params$sb_lvar/2))
hist(exp(lsb_samp))

s_sq <- get_s_sq(nn_model$fc1)  
ztilsq <- get_ztil_sq(nn_model$fc1)
round(ztilsq, 3)
k1 <- get_kappas(nn_model$fc1)


k2 <- get_kappas(nn_model$fc2)
s_sq2 <- get_s_sq(nn_model$fc2)
ztil_sq2 <- get_ztil_sq(nn_model$fc2)
zsq2 <- s_sq2 * ztil_sq2
round(ztil_sq2, 3)
round(zsq2, 3)
gm_zsq2 <- exp(mean(log(zsq2)))


get_zsq <- function(nn_model_layer, ln_fcn = ln_mode){
  s_sq <- get_s_sq(nn_model_layer, ln_fcn)
  ztil_sq <- get_ztil_sq(nn_model_layer, ln_fcn)
  return(s_sq * ztil_sq)
}

geom_mean <- function(vec){
  exp(mean(log(vec)))
}
s_sq3 <- get_s_sq(nn_model$fc3)
zsq3 <- get_zsq(nn_model$fc3)
gm_zsq3 <- geom_mean(zsq3)

s_sq
s_sq2
s_sq3


gm_zsq2

# dividing by successive layers' s_sq params --- idea is that 
# successive layers' global shrinkage parameters indicates weights that should have been
# removed but weren't.  Since the previous layer has more neurons than it should, this 
# makes the previous layer's global scale factor smaller than it should be.
sq1_corrected <- s_sq / (s_sq2 * s_sq3)
k1_corrected <- 1 / (1 + ztilsq*sq1_corrected)
round(k1_corrected, 2)
# ... Actually, this argument seems better applied when taking into account the 
# local scale factors (since the local scale factors are what actually dictate whether
# a neuron from the revious layer is removed).


#### workable / justifiable? ---- 
# Divide current layer's scale parameters by the geometric mean 
# of successive layers' scale params --- idea is that 
# successive layer's scale params indicates neurons that 
# **should have been removed** from the previous layer
# but weren't.  Since the previous layer has more neurons than it should, this 
# makes the previous layer's global scale factor smaller than it should be.
z_sq <- get_zsq(nn_model$fc1)
zsq2 <- get_zsq(nn_model$fc2)
gm_zsq2 <- geom_mean(zsq2)
zsq3 <- get_zsq(nn_model$fc3)
gm_zsq3 <- geom_mean(zsq3)


mod_zsq1 <- z_sq / gm_zsq2
mod_k1 <- 1 / (1 + mod_zsq1)
round(mod_k1, 2)
round(k1, 3)



Wz_params <- get_Wz_params(nn_model$fc1)
Wz_mu <- Wz_params$Wz_mu
Wz_var <- Wz_params$Wz_var

## look at results from all trained mods
k_corr_mat <- matrix(NA, nrow = length(seeds), ncol = 104)
err_mat <- matrix(NA, nrow = length(seeds), ncol = 6)
for (i in 1:length(seeds)){
  nn_mod <- torch_load(mod_fnames[i])
  s_sq1 <- get_s_sq(nn_mod$fc1)
  s_sq2 <- get_s_sq(nn_mod$fc2)
  s_sq3 <- get_s_sq(nn_mod$fc3)
  ztil_sq1 <- get_ztil_sq(nn_mod$fc1)
  ztil_sq2 <- get_ztil_sq(nn_mod$fc2)
  ztil_sq3 <- get_ztil_sq(nn_mod$fc3)
  ztil_sq_gm2 <- geom_mean(ztil_sq2)
  ztil_sq_gm3 <- geom_mean(ztil_sq3)
  k_corr <- (1 + ztil_sq1 * s_sq1 / s_sq2)^-1
  k_corr <- (1 + ztil_sq1 * s_sq1)^-1
  err <- err_from_dropout(dropout_vec = k_corr, max_bfdr = 0.05)
  err_mat[i, ] <- err
  k_corr_mat[i, ] <- k_corr
}
colnames(err_mat) <- names(err)
err_mat
round(k_corr_mat[, 1:10], 2)

round(apply(x, 2, var), 2)
y <- as_array(simdat$y)
var(y)




# counterargument: more layers 


mod2_zsq1 <- z_sq / gm_zsq2
round(mod2_zsq1, 2)
mod2_k1 <- 1 / (1 + mod2_zsq1)
round(mod2_k1, 2)
round(k1, 3)








# REDUCED model ----
# reduced layers # neurons in both layers 1 and 2
# this is not trained enough, most likely, and I didn't include all params by mistake
fname_stem <- here::here(
  "sims", "results", 
  "fcnl_hshoe_mod_12500obs_191578_RED"
)
nn_model <- torch::torch_load(
  paste0(fname_stem, ".pt"), 
  device = "cpu")

load(paste0(fname_stem, ".RData"))

get_s_sq(nn_model$fc1)  #### ugh, this is even smaller than in the non-reduced model.
ztilsq <- get_ztil_sq(nn_model$fc1)
round(ztilsq, 3)
round(1 / (1 + ztilsq), 3)




#### check lm ----

lm_bin_err_mat <- matrix(NA, nrow = length(seeds), ncol = 4)
lm_varsel <- matrix(NA, nrow = length(seeds), ncol = 104)
for (i in 1:length(seeds)){
  seed <- seeds[i]
  set.seed(seed)
  torch_manual_seed(seed)
  
  # load to get original simulation params sim_res$sim_params
  load(res_fnames[i]) 
  
  # (re)generate data
  simdat <- sim_func_data(
    n_obs = 1250,
    d_in = sim_res$sim_params$d_in,
    flist = sim_res$sim_params$flist,
    err_sigma = sim_res$sim_params$err_sig
  )
  # if (sim_params$use_cuda){
  #   simdat$x <- simdat$x$to(device = "cuda")
  #   simdat$y <- simdat$y$to(device = "cuda")
  # }
  
  # convert torch tensors to R objects
  x <- as_array(simdat$x)
  y <- as_array(simdat$y)
  
  lm_fit <- lm(y ~ x)
  pvals <- summary(lm_fit)$coef[-1, 4]
  lm_delta_i <- p.adjust(pvals, method = "BH", n = length(pvals)) < 0.05
  lm_varsel[i, ] <- lm_delta_i
  lm_bin_err <- binary_err_rate(est = lm_delta_i, tru = c(rep(1, 4), rep(0, 100)))
  lm_bin_err_mat[i, ] <- lm_bin_err
}

colnames(lm_bin_err_mat) <- names(lm_bin_err)
lm_bin_err_mat
lm_varsel

plot_datagen_fcns(sim_res$sim_params$flist)
# as we would expect, lm picks up on functions that have some overall slope



# spike-slab ----
library(BoomSpikeSlab)
ss_bin_err_mat <- matrix(NA, nrow = length(seeds), ncol = 4)
for (i in 1:length(seeds)){
  seed <- seeds[i]
  set.seed(seed)
  torch_manual_seed(seed)
  
  # load to get original simulation params sim_res$sim_params
  load(res_fnames[i]) 
  
  # (re)generate data
  simdat <- sim_func_data(
    n_obs = 1250,
    d_in = sim_res$sim_params$d_in,
    flist = sim_res$sim_params$flist,
    err_sigma = sim_res$sim_params$err_sig
  )
  # if (sim_params$use_cuda){
  #   simdat$x <- simdat$x$to(device = "cuda")
  #   simdat$y <- simdat$y$to(device = "cuda")
  # }
  
  # convert torch tensors to R objects
  x <- as_array(simdat$x)
  y <- as_array(simdat$y)
  
  
  ## spike-slab ----
  
  # need to include intercept in covariate matrix given to IndependentSpikeSlabPrior()
  modmat <- cbind(1, x)
  prior = IndependentSpikeSlabPrior(modmat, y, 
                                    expected.model.size = 20,
                                    prior.beta.sd = rep(1, ncol(modmat))) 
  
  lm_ss = lm.spike(y ~ x, niter = 1000, prior = prior, ping = 0)
  summary(lm_ss)$coef
  
  # reorder results to make vars appear in original order 
  # rather than ordered by Marginal Posterior Inclusion Probability
  ss_allcoefs_unordered <- summary(lm_ss)$coef
  ss_intercept <- ss_allcoefs_unordered[rownames(ss_allcoefs_unordered)=="(Intercept)",]
  ss_coefs_unordered <- ss_allcoefs_unordered[rownames(ss_allcoefs_unordered)!="(Intercept)",]
  coef_order <- as.numeric(sapply(strsplit(rownames(ss_coefs_unordered), "x"), function(x) x[2]))
  ss_allcoefs <- rbind(
    "(Intercept)" = ss_intercept,
    ss_coefs_unordered[order(coef_order),]
  )
  
  ss_median_mod <- ss_allcoefs[, 5] > 0.5
  
  ss_bin_err <- binary_err_rate(est = ss_median_mod[-1], tru = c(rep(1, 4), rep(0, 100)))
  ss_bin_err_mat[i, ] <- ss_bin_err
}
colnames(ss_bin_err_mat) <- names(ss_bin_err)
ss_bin_err_mat







# calibrate prior on tau: ----
# Piironen & Vehtari 2017 "Sparsity Info & Regularization in the HShoe" (and others)

calibrate_tau <- function(p_0, J, sig, nobs){
  p_0 / (J - p_0) * sig / sqrt(nobs)
}

calibrate_tau(p_0 = 4, J = 104, sig = 1, nobs = 10000)

# regularizes hshoe: ----
lambda_reg <- function(lambda, tau, c){
  (c^2 * lambda^2) / (c^2 + tau^2 * lambda^2)
}

s_sq1 <- get_s_sq(nn_mod$fc1)
ztil_sq1 <- get_ztil_sq(nn_mod$fc1)

ztil_sq1_reg <- lambda_reg(lambda = sqrt(ztil_sq1), tau = sqrt(s_sq1), c=2)
round(cbind(ztil_sq1, ztil_sq1_reg), 3)[1:10, ]


zsq <- s_sq1 * ztil_sq1

k1 <- (1 + zsq)^-1
m_eff <- 16 * sum(zsq / (1 + zsq))



#### corrected estimates of kappa:

fname_stem <- here::here(
  "sims", "results", 
  "fcnl_hshoe_mod_12500obs_191578"
)
mod <- torch::torch_load(
  paste0(fname_stem, ".pt"), 
  device = "cpu")

load(paste0(fname_stem, ".RData"))
tail(sim_res$loss_mat)


# estimating actual shrinkage in layer 1 from all 3 layers:
#   (1 - K1) * Frobenius norm of || (1-K2)(W2)(1-K3)(W3) ||
f_norm <- function(M){
  sqrt(sum(M*M))
}
v_norm <- function(V){
  sqrt(sum(V*V))
}

k2 <- get_kappas(mod$fc2)
k3 <- get_kappas(mod$fc3)
W1 <- as_array(mod$fc1$weight_mu)
W2 <- as_array(mod$fc2$weight_mu)
W3 <- as_array(mod$fc3$weight_mu)

K2 <- diag(1-k2)
K3 <- diag(1-k3)



correction <- v_norm(K2 %*% t(W2) %*% K3 %*% t(W3))
correction2 <- 16 * correction

correction3 <- v_norm(K2) * f_norm(W2) * v_norm(K3) * f_norm(W3)

tau_sq <- get_s_sq(mod$fc1)
ztil_sq <- get_ztil_sq(mod$fc1)
k1 <- (1 + tau_sq*ztil_sq)^(-1)

z_sq_corrected <- ztil_sq * tau_sq * 104
corrected_k1 <- (1 + z_sq_corrected)^(-1)
round(corrected_k1, 2)
round(k1, 2)[1:10]

