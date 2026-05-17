





#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
# modified forward portion of torch_horseshoe_klcorrected
source(here("Rcode", "torch_horseshoe_smallbias.R")) 
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))



get_smallest_msediff_epoch <- function(lmat){
  tetr_diff <- abs(lmat[, 3] - lmat[, 2])
  smallest_row <- which(tetr_diff == min(tetr_diff))
  if (length(smallest_row) == 0){
    smallest_row <- nrow(lmat)
  } else if (length(smallest_row) > 1){
    smallest_row <- smallest_row[length(smallest_row)]
  }
  smallest_row
}


get_smallest_kl_epoch <- function(lmat){
  kls <- lmat[,4]
  smallest_row <- which(kls == min(kls))
  if (length(smallest_row) == 0){
    smallest_row <- nrow(lmat)
  } else if (length(smallest_row) > 1){
    smallest_row <- smallest_row[length(smallest_row)]
  }
  smallest_row
}

get_smallest_testmse_epoch <- function(lmat){
  te_mses <- lmat[,3]
  smallest_row <- which(te_mses == min(te_mses))
  if (length(smallest_row) == 0){
    smallest_row <- nrow(lmat)
  } else if (length(smallest_row) > 1){
    smallest_row <- smallest_row[length(smallest_row)]
  }
  smallest_row
}


f1_calc <- function(fdr, tpr){
  prec <- 1-fdr
  if (prec + tpr == 0){return(0)}
  2 * (prec*tpr) / (prec + tpr)
}

 
metrics_err_by_max_bfdr <- function(dropout_vec, true_vec, bfdr_vec){
  # add f1 score:
  # precision = 1-fdr
  # f1 = 2 * (precision*recall)/(precision + recall)
  err_mat <- err_by_max_bfdr(dropout_vec, true_vec, bfdr_vec)$err_mat
  f1 <- apply(err_mat, 1, function(X) f1_calc(fdr=X[2], tpr = X[5]))
  res <- cbind(err_mat, f1)
  colnames(res) <- c("max_bfdr", "fdr", "bfdr", "FPR", "TPR_sens_recall", "FNR", "TN_specificity", "f1")
  res
}


# ORIG FCNS:       compiled results                                       sim stem
# hshoe2det4_1k_50sims_origfcns_bfdr_arr.Rdata       100 sims available:   nfdsmallbias_mutcorr0.5_5x161000obs_
# hshoe2det4_2k_50sims_origfcns_bfdr_arr.Rdata       50 sims available:    nfdsmallbias_mutcorr0.5_5x162000obs_
# hshoe2det4_5k_50sims_origfcns_bfdr_arr.Rdata       125 sims available:   nfdsmallbias_mutcorr0.5_5x165000obs_
#
# hshoe4det1_1k_50sims_origfcns_bfdr_arr.Rdata       66 available          hshoesmallbias_mutcorr0.5_5x161000obs_
# hshoe4det1_2k_50sims_origfcns_bfdr_arr.Rdata       50 available          hshoesmallbias_mutcorr0.5_5x162000obs_
# hshoe4det1_5k_50sims_origfcns_bfdr_arr.Rdata       34 avilable           hshoesmallbias_mutcorr0.5_5x165000obs_


# MOD FCNS
# compiled/hshoe2det4_1k_50sims_modfcns_bfdr_arr.Rdata        64 available:         meanfssmallbias_5x16_origmodsupint_p100_mcor.5_1000obs_
# compiled/hshoe2det4_2k_50sims_modfcns_bfdr_arr.Rdata        0 available:          meanfssmallbias_5x16_origmodsupint_p100_mcor.5_2000obs_
# compiled/hshoe2det4_5k_50sims_modfcns_bfdr_arr.Rdata        52 available:         meanfssmallbias_5x16_origmodsupint_p100_mcor.5_5000obs_

# hshoe4det1


stem <- here::here("final_sims", "results", "meanfshshoesmallbias_5x16_origmodsupint_p100_mcor.5_2000obs_")
modfcns_TF <- grepl("meanfs", stem)
n_sims = 50 
max_bfdrs <- c(0.01, 0.05, 0.1, 0.25)

if (modfcns_TF){
  true_vec <- rep(0, 108)
  true_vec[1:8] <- 1
  fname_suffix <- "modfcns_bfdr_arr"   
} else {
  true_vec <- rep(0, 104)
  true_vec[1:4] <- 1
  fname_suffix <- "origfcns_bfdr_arr"
}

# started off some with 5, some with 10.  Figure out which ones have 5, vs 10
overall_seeds <- as.numeric(c(516, paste0(516, 0:13)))
possible_sim_seeds <- c()

for (i in 1:length(overall_seeds)){
  set.seed(overall_seeds[i])
  possible_sim_seeds <- c(possible_sim_seeds, floor(runif(n = 10, 0, 1000000)))
}

# which of these files exist (some simulations stopped b/c of computer issues)
poss_fnames <- paste0(stem, possible_sim_seeds, ".RData")
exists_TF <- 
  has_corrections_by_layer <- rep(F, length(poss_fnames))

for (i in 1:length(poss_fnames)){
  exists_TF[i] <- file.exists(poss_fnames[i])
}

sum(exists_TF)
length(unique(poss_fnames[exists_TF]))
if (length(unique(poss_fnames[exists_TF])) < n_sims) {warning("fewer than ", n_sims, " exist")}


sim_fnames <- paste0(stem, possible_sim_seeds[exists_TF], ".RData")[1:n_sims]
mod_fnames <- paste0(stem, possible_sim_seeds[exists_TF], ".pt")[1:n_sims]

# construct filename ----
load(sim_fnames[1])
nn_mod <- torch_load(paste0(stem, possible_sim_seeds[exists_TF][1], ".pt"))
hshoe_layers <- grepl("fc", names(nn_mod$children))
det_layers <- grepl("det", names(nn_mod$children))
architecture_str <- paste0("hshoe", sum(hshoe_layers), "det", sum(det_layers))

compiled_stem <- paste0(
  architecture_str, "_", 
  round(sim_res$sim_params$n_obs/1000), "k_",
  n_sims, "sims_", fname_suffix
)
compiled_fname <- here::here("final_sims", "compiled", paste0(compiled_stem, ".Rdata"))


# store results
ksn50k_mat <- 
  ksntc50k_mat <- 
  ksn_metrictest_mat <- 
  ksntc_metrictest_mat <-
  ktc50k_mat <- 
  ktc_metrictest_mat <- matrix(NA, nrow = length(sim_fnames), ncol = ncol(sim_res$kappa_mat))

perf_mat <- matrix(NA, nrow = n_sims, ncol = 4)
colnames(perf_mat) <- c("train_sig", "mse_train", "mse_test", "kl")
perf_mat_metrictest <- perf_mat

# quick check of sn and sntc corrected kappas at last epoch
metrictest_rows <- rep(NA, length(sim_fnames))
for (f_ind in 1:length(sim_fnames)){
  load(sim_fnames[f_ind])
  last_epoch <- nrow(sim_res$kappa_sn_mat)
  perf_mat[f_ind, ] <- c(sim_res$sim_params$train_sig, sim_res$loss_mat[last_epoch, 2:4])
  
  ksn50k_mat[f_ind, ] <- sim_res$kappa_sn_mat[last_epoch,]
  ksntc50k_mat[f_ind, ] <- sim_res$kappa_sntc_mat[last_epoch,]
  ktc50k_mat[f_ind, ] <- sim_res$kappa_tc_mat[last_epoch,]
  
  row_ind <- get_smallest_testmse_epoch(sim_res$loss_mat)
  perf_mat_metrictest[f_ind, ] <- c(sim_res$sim_params$train_sig, sim_res$loss_mat[row_ind, 2:4])
  metrictest_rows[f_ind] <- row_ind
  ksn_metrictest_mat[f_ind, ] <- sim_res$kappa_sn_mat[row_ind,]
  ksntc_metrictest_mat[f_ind, ] <- sim_res$kappa_sntc_mat[row_ind,]
  ktc_metrictest_mat[f_ind, ] <- sim_res$kappa_tc_mat[last_epoch,]
}

# sum(ksn50k_mat[, 1:4] < 0.05)     # 1000obs: 126 out of 300    # 5000obs: 206 
# sum(ksn50k_mat[, 5:104] < 0.5)    # 1000obs: no FPs            # 5000obs: 0
# 
# sum(ksntc50k_mat[, 1:4] < 0.05)   # 1000obs: 191 out of 300    # 5000obs: 351
# sum(ksntc50k_mat[, 5:104] < 0.5)  # 1000obs: no FPs            # 5000obs: 100

kmat_list <- list(
  "ktc50k_mat" = ktc50k_mat,
  "ksn50k_mat" = ksn50k_mat,
  "ksntc50k_mat" = ksntc50k_mat,
  "ktc_metrictest_mat" = ktc_metrictest_mat,
  "ksn_metrictest_mat" = ksn_metrictest_mat,
  "ksntc_metrictest_mat" = ksntc_metrictest_mat
)


# metrictest_rows
sim_res$loss_mat[metrictest_rows,5] # KLweight at best test_mse
sum(ksn_metrictest_mat[, 1:4] < 0.05) # 17 out of 300          # 5000obs: 25
sum(ksn_metrictest_mat[, 5:ncol(sim_res$kappa_mat)] < 0.5)  # no FPs               # 5000obs: 0

sum(ksntc_metrictest_mat[, 1:4] < 0.05) # 156 out of 300       # 5000obs: 313
sum(ksntc_metrictest_mat[, 5:ncol(sim_res$kappa_mat)] < 0.5)  # 8                  # 5000obs: 11



# ksn50k_bfdr0.01 <- matrix(NA, nrow = , ncol = 8)
# colnames(ksn50k_bfdr0.01) <- c("max_bfdr", "fdr", "bfdr", "FPR", "TPR_sens_recall", "FNR", "TN_specificity", "f1")
# 
# ksn50k_bfdr0.05 <- 
#   ksn50k_bfdr0.1 <- 
#   ksn50k_bfdr0.25 <- 
#   ksntc50k_bfdr0.01 <- 
#   ksntc50k_bfdr0.05 <- 
#   ksntc50k_bfdr0.1 <- 
#   ksntc50k_bfdr0.25 <- 
#   ksn_metrictest_bfdr0.01 <- 
#   ksn_metrictest_bfdr0.05 <- 
#   ksn_metrictest_bfdr0.1 <- 
#   ksn_metrictest_bfdr0.25 <-
#   ksntc_metrictest_bfdr0.01 <- 
#   ksntc_metrictest_bfdr0.05 <- 
#   ksntc_metrictest_bfdr0.1 <- 
#   ksntc_metrictest_bfdr0.25 <-ksn50k_bfdr0.01

ksn50k_bfdr_arr <- array(NA, dim = c(nrow(ksn50k_mat), 8, length(max_bfdrs)))
errmat_colnames <- c("max_bfdr", "fdr", "bfdr", "FPR", "TPR_sens_recall", "FNR", "TN_specificity", "f1")
dimnames(ksn50k_bfdr_arr) <- list(paste0("sim_", 1:nrow(ksn50k_mat)), errmat_colnames, paste0("bfdr_",max_bfdrs))

ksntc50k_bfdr_arr <- 
  ksn_metrictest_bfdr_arr <-
  ksntc_metrictest_bfdr_arr <- 
  ktc50k_bfdr_arr <-
  ktc_metrictest_bfdr_arr <- ksn50k_bfdr_arr

for (k in 1:nrow(ksn50k_mat)){
  err_mat <- metrics_err_by_max_bfdr(dropout_vec = ksn50k_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ksn50k_bfdr_arr[k, , ] <- t(err_mat)
  # ksn50k_bfdr0.01[k, ] <- c(err_mat[1, ])
  # ksn50k_bfdr0.05[k, ] <- c(err_mat[2, ])
  # ksn50k_bfdr0.1[k, ] <- c(err_mat[3, ])
  # ksn50k_bfdr0.25[k, ] <- c(err_mat[4, ])
  
  err_mat_sntc <- metrics_err_by_max_bfdr(dropout_vec = ksntc50k_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ksntc50k_bfdr_arr[k, , ] <- t(err_mat_sntc)
  # ksntc50k_bfdr0.01[k, ] <- c(err_mat_sntc[1, ])
  # ksntc50k_bfdr0.05[k, ] <- c(err_mat_sntc[2, ])
  # ksntc50k_bfdr0.1[k, ] <- c(err_mat_sntc[3, ])
  # ksntc50k_bfdr0.25[k, ] <- c(err_mat_sntc[4, ])
  
  err_mat_tc <- metrics_err_by_max_bfdr(dropout_vec = ktc50k_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ktc50k_bfdr_arr[k, , ] <- t(err_mat_tc)
  
  err_mat_tc_metrictest <- metrics_err_by_max_bfdr(dropout_vec = ktc_metrictest_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ktc_metrictest_bfdr_arr[k, , ] <- t(err_mat_tc_metrictest)

  err_mat_metrictest <- metrics_err_by_max_bfdr(dropout_vec = ksn_metrictest_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ksn_metrictest_bfdr_arr[k, , ] <- t(err_mat_metrictest)
  # ksn_metrictest_bfdr0.01[k, ] <- c(err_mat_sntc[1, ])
  # ksn_metrictest_bfdr0.05[k, ] <- c(err_mat_sntc[2, ])
  # ksn_metrictest_bfdr0.1[k, ] <- c(err_mat_sntc[3, ])
  # ksn_metrictest_bfdr0.25[k, ] <- c(err_mat_sntc[4, ])
  
  err_mat_metrictest <- metrics_err_by_max_bfdr(dropout_vec = ksntc_metrictest_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ksntc_metrictest_bfdr_arr[k, , ] <- t(err_mat_metrictest)
  # ksntc_metrictest_bfdr0.01[k, ] <- c(err_mat_sntc[1, ])
  # ksntc_metrictest_bfdr0.05[k, ] <- c(err_mat_sntc[2, ])
  # ksntc_metrictest_bfdr0.1[k, ] <- c(err_mat_sntc[3, ])
  # ksntc_metrictest_bfdr0.25[k, ] <- c(err_mat_sntc[4, ])
}

bfdr_arr_list <- list(
  "perf_mat" = perf_mat,
  "perf_mat_metrictest" = perf_mat_metrictest,
  "ktc50k_bfdr_arr" = ktc50k_bfdr_arr,
  "ksn50k_bfdr_arr" = ksn50k_bfdr_arr,
  "ksntc50k_bfdr_arr" = ksntc50k_bfdr_arr,
  "ktc_metrictest_bfdr_arr" = ktc_metrictest_bfdr_arr,
  "ksn_metrictest_bfdr_arr" = ksn_metrictest_bfdr_arr,
  "ksntc_metrictest_bfdr_arr" = ksntc_metrictest_bfdr_arr
)

res <- list(kmat_list, bfdr_arr_list)
save(res, file = compiled_fname)

cat_color(paste0("results saved to ", compiled_fname))


load(compiled_fname)



t(apply(ktc50k_bfdr_arr, c(2, 3), mean))
t(apply(ktc50k_bfdr_arr, c(2, 3), sd))

t(apply(ksn50k_bfdr_arr, c(2, 3), mean))
t(apply(ksn50k_bfdr_arr, c(2, 3), sd))

t(apply(res[[2]]$ksntc50k_bfdr_arr, c(2, 3), mean))
t(apply(res[[2]]$ksntc50k_bfdr_arr, c(2, 3), sd))


t(apply(ktc_metrictest_bfdr_arr, c(2, 3), mean))
t(apply(ktc_metrictest_bfdr_arr, c(2, 3), sd))

t(apply(ksn_metrictest_bfdr_arr, c(2, 3), mean))
t(apply(ksn_metrictest_bfdr_arr, c(2, 3), sd))

t(apply(ksntc_metrictest_bfdr_arr, c(2, 3), mean))
t(apply(ksntc_metrictest_bfdr_arr, c(2, 3), sd))



# #### plot ROC
# 
# load(here("final_sims", "compiled", "hshoe2det4_1k_50sims_origfcns_bfdrall_arr.Rdata"))
# 
# mat <- res[[2]]$ksntc50k_bfdr_arr
# #
# res[[2]]$ksntc50k_bfdr_arr[,,2]
# 
# # column 4 is FPR, col 5 is TPR
# FPRs <- apply(res[[2]]$ksntc50k_bfdr_arr, 3, function(X) X[,4])
# TPRs <- apply(res[[2]]$ksntc50k_bfdr_arr, 3, function(X) X[,5])
# fdrs <- apply(res[[2]]$ksntc50k_bfdr_arr, 3, function(X) X[,2])
# bfdrs <- apply(res[[2]]$ksntc50k_bfdr_arr, 3, function(X) X[,3])
# 
# 
# 
# 
# FPR_mean <- apply(FPRs, 2, mean)
# TPR_mean <- apply(TPRs, 2, mean)
# FPR_sd <- apply(FPRs, 2, sd)
# TPR_sd <- apply(TPRs, 2, sd)
# 
# fdr_mean <- apply(fdrs, 2, mean)
# bfdr_mean <- apply(bfdrs, 2, mean)
# fdr_sd <- apply(fdrs, 2, sd)
# bfdr_sd <- apply(bfdrs, 2, sd)
# plot(fdr_mean ~ bfdr_mean)
# 
# bfdr_qtiles <-t(apply(bfdrs, 2, function(X) quantile(X, c(0.25, 0.975))))
# fdr_qtiles <- t(apply(fdrs, 2, function(X) quantile(X, c(0.25, 0.975))))
# fpr_qtiles <- t(apply(FPRs, 2, function(X) quantile(X, c(0.25, 0.975))))
# tpr_qtiles <- t(apply(TPRs, 2, function(X) quantile(X, c(0.25, 0.975))))
# 
# bfdr_minmax <-t(apply(bfdrs, 2, range))
# fdr_minmax <- t(apply(fdrs, 2, range))
# fpr_minmax <- t(apply(FPRs, 2, range))
# tpr_minmax <- t(apply(TPRs, 2, range))
# 
# 
# 
# 
# round(t(fdrs[,50:60 ]), 2)
# max_bfdrs <- 0:100/100
# roc_df <- data.frame(
#   "bfdr_threshs" = max_bfdrs,
#   "FPR_mean" = FPR_mean,
#   "TPR_mean" = TPR_mean,
#   "FPR_lo" = fpr_qtiles[,1],
#   "FPR_hi" = fpr_qtiles[,2],
#   "TPR_lo" = tpr_qtiles[,1],
#   "TPR_hi" = tpr_qtiles[,2],
#   "fdr_mean" = fdr_mean,
#   "bfdr_mean" = bfdr_mean,
#   "fdr_lo" = fdr_minmax[, 1],
#   "fdr_hi" = fdr_minmax[, 2],
#   "bfdr_lo" = bfdr_minmax[, 1],
#   "bfdr_hi" = bfdr_minmax[, 2]
# )
# 
# cbind(roc_df$bfdr_threshs, roc_df$fdr_hi)
# 
# 
# library(latex2exp)
# roc1k <- roc_df %>% 
#   ggplot() + 
#   geom_line(aes(y = TPR_mean, x = FPR_mean)) + 
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed") + 
#   coord_fixed(ratio = 1) + 
#   theme(aspect.ratio = 1) +
#   scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
#   scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
#   labs(subtitle = TeX("ROC curve for HS-2, $n=1000$"),
#        y = "mean true positive rate",
#        x = "mean false positive rate")
# roc1k
# ggsave(roc1k, filename = here("final_sims", "figs", "roc_hs21k.png"))
# cbind(TPR_mean, FPR_mean)
# 
# bfdr_calibration <- roc_df %>% 
#   ggplot() + 
#   geom_line(
#     aes(
#       x = bfdr_threshs, y = bfdr_mean
#     ), color = "blue"
#   ) + 
#   geom_ribbon(
#     aes(
#       x = bfdr_threshs, ymax = bfdr_hi, ymin = bfdr_lo
#     ), fill = "blue", alpha = 0.2
#   ) + 
#   geom_line(
#     aes(
#       x = bfdr_threshs, y = fdr_mean
#     ), color = "red"
#   ) + 
#   geom_ribbon(
#     aes(
#       x = bfdr_threshs, ymax = fdr_hi, ymin = fdr_lo
#     ), fill = "red", alpha = 0.2
#   ) + 
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed") + 
#   labs(
#     subtitle = "BFDR calibration: BFDR (blue), FDR (red)",
#     x = "nominal BFDR maximum",
#     y = "realized BFDR/FDR"
#   )
# bfdr_calibration
# ggsave(bfdr_calibration, filename = here("final_sims", "figs", "bfdr_calib.png"))
# 




