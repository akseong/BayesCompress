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
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "torch_horseshoe_initvals.R"))
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









# RETRAIN ----
seednum <- 2
nn_model <- torch::torch_load(mod_fnames[seednum], device = "cpu")
load(res_fnames[seednum])
sim_res$loss_mat
sim_params <- sim_res$sim_params
sim_params$use_cuda <- nn_model$fc1$atilde_logvar$is_cuda

set.seed(seeds[seednum])
torch_manual_seed(seeds[seednum])
simdat <- sim_func_data(
  n_obs = sim_params$n_obs, 
  d_in = sim_params$d_in, 
  flist = sim_params$flist, 
  err_sigma = sim_params$err_sig,
)
if (nn_model$fc1$atilde_logvar$is_cuda){
  simdat$x <- simdat$x$to(device = "cuda")
  simdat$y <- simdat$x$to(device = "cuda")
}

## visualize ----
pred_mats <- make_pred_mats(
  flist = sim_params$flist,
  d_in = sim_params$d_in
)

fcn_plt <- plot_datagen_fcns(
  flist = sim_params$flist,
  min_x = -5,
  max_x = 5
)

plot_fcn_preds(torchmod = nn_model, pred_mats)


## selecting nodes with dropout probability < mean (mean-ish model?) ----
k1 <- get_kappas(nn_model$fc1)
round(k1, 3)

k2 <- get_kappas(nn_model$fc2)
round(k2, 3)
l1_selected_nodes <- k2 < mean(k2)
d1_red <- sum(l1_selected_nodes)
d1_red

k3 <- get_kappas(nn_model$fc3)
round(k3, 3)  
l2_selected_nodes <- k3 < mean(k3)
d2_red <- sum(l2_selected_nodes)
d2_red


# define model, initialize params:
## define model
MLHS_red <- nn_module(
  "MLHS_red",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = sim_params$d_in, 
      out_features = d1_red,
      use_cuda = sim_params$use_cuda,
      tau = 1,
      init_weight = nn_model$fc1$weight_mu[l1_selected_nodes, ],
      init_bias = nn_model$fc1$bias_mu[l1_selected_nodes],
      init_sa = nn_model$fc1$sa_mu, 
      init_sb = nn_model$fc1$sb_mu, 
      init_atilde = nn_model$fc1$atilde_mu, 
      init_btilde = nn_model$fc1$btilde_mu, 
      init_weight_logvar = nn_model$fc1$weight_logvar[l1_selected_nodes, ], 
      init_bias_logvar = nn_model$fc1$bias_logvar[l1_selected_nodes], 
      init_sa_logvar = nn_model$fc1$sa_logvar, 
      init_sb_logvar = nn_model$fc1$sb_logvar, 
      init_atilde_logvar = nn_model$fc1$atilde_logvar, 
      init_btilde_logvar = nn_model$fc1$btilde_logvar, 
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc2 = torch_hs(
      in_features = sim_params$d_hidden1,
      out_features = d2_red,
      use_cuda = sim_params$use_cuda,
      tau = 1,
      init_weight = nn_model$fc2$weight_mu[l2_selected_nodes, l1_selected_nodes],
      init_bias = nn_model$fc2$bias_mu[l2_selected_nodes],
      init_sa = nn_model$fc2$sa_mu, 
      init_sb = nn_model$fc2$sb_mu, 
      init_atilde = nn_model$fc2$atilde_mu[l1_selected_nodes], 
      init_btilde = nn_model$fc2$btilde_mu[l1_selected_nodes], 
      init_weight_logvar = nn_model$fc2$weight_logvar[l2_selected_nodes, l1_selected_nodes], 
      init_bias_logvar = nn_model$fc2$bias_logvar[l2_selected_nodes], 
      init_sa_logvar = nn_model$fc2$sa_logvar, 
      init_sb_logvar = nn_model$fc2$sb_logvar, 
      init_atilde_logvar = nn_model$fc2$atilde_logvar[l1_selected_nodes], 
      init_btilde_logvar = nn_model$fc2$btilde_logvar[l1_selected_nodes], 
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc3 = torch_hs(
      in_features = d2_red,
      #   out_features = sim_params$d_hidden3,
      #   use_cuda = sim_params$use_cuda,
      #   tau = 1,
      #   init_weight = NULL,
      #   init_bias = NULL,
      #   init_alpha = 0.9,
      #   clip_var = TRUE
      # )
      # 
      # self$fc4 = torch_hs(
      #   in_features = sim_params$d_hidden3,
      #   out_features = sim_params$d_hidden4,
      #   use_cuda = sim_params$use_cuda,
      #   tau = 1,
      #   init_weight = NULL,
      #   init_bias = NULL,
      #   init_alpha = 0.9,
      #   clip_var = TRUE
      # )
      # 
      # self$fc5 = torch_hs(
      #   in_features = sim_params$d_hidden4,
      out_features = sim_params$d_out,
      use_cuda = sim_params$use_cuda,
      tau = 1,
      init_weight = nn_model$fc3$weight_mu[, l2_selected_nodes],
      init_bias = nn_model$fc3$bias_mu,
      init_sa = nn_model$fc3$sa_mu, 
      init_sb = nn_model$fc3$sb_mu, 
      init_atilde = nn_model$fc3$atilde_mu[l2_selected_nodes], 
      init_btilde = nn_model$fc3$btilde_mu[l2_selected_nodes], 
      init_weight_logvar = nn_model$fc3$weight_logvar[, l2_selected_nodes], 
      init_bias_logvar = nn_model$fc3$bias_logvar, 
      init_sa_logvar = nn_model$fc3$sa_logvar, 
      init_sb_logvar = nn_model$fc3$sb_logvar, 
      init_atilde_logvar = nn_model$fc3$atilde_logvar[l2_selected_nodes], 
      init_btilde_logvar = nn_model$fc3$btilde_logvar[l2_selected_nodes], 
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
  },
  
  forward = function(x) {
    x %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      nnf_relu() %>%
      self$fc3() # %>%
    # nnf_relu() %>%
    # self$fc4() %>% 
    # nnf_relu() %>%
    # self$fc5()
  },
  
  
  
  get_model_kld = function(){
    kl1 = self$fc1$get_kl()
    kl2 = self$fc2$get_kl()
    kl3 = self$fc3$get_kl()
    # kl4 = self$fc3$get_kl()
    # kl5 = self$fc3$get_kl()
    kld = kl1 + kl2 + kl3 #+ kl4 + kl5
    return(kld)
  }
)

sim_params$model_red <- MLHS_red
sim_params$sim_name <- paste0("reduced model based on kappas from: ", sim_params$sim_name)
sim_params$d_hidden1 <- d1_red
sim_params$d_hidden2 <- d2_red
save_mod_path_stem <- here::here("sims", 
                                 "results", 
                                 paste0("fcnl_hshoe_mod_", 
                                        sim_params$n_obs, "obs_",
                                        seeds[seednum],
                                        "_RED"
                                 ))

sim_params$train_epochs <- 2e4
sim_params$report_every <- 1000

sim_res_red <- sim_hshoe(
  seed = seeds[seednum],
  sim_ind = NULL,
  sim_params,     # same as before, but need to include flist
  nn_model = MLHS_red,   # torch nn_module,
  verbose = TRUE,   # provide updates in console
  want_plots = FALSE,   # provide graphical updates of KL, MSE
  want_fcn_plots = TRUE,   # display predicted functions
  save_fcn_plots = FALSE,
  want_all_params = FALSE,
  local_only = FALSE,
  save_mod = TRUE,
  save_results = TRUE,
  save_mod_path_stem = save_mod_path_stem
)










