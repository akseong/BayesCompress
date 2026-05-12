





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
# hshoe2det3_origfns_1k_50sims_compiled.RData       100 sims available:   nfdsmallbias_mutcorr0.5_5x161000obs_
# hshoe2det3_origfns_2k_50sims_compiled.RData       50 sims available:    nfdsmallbias_mutcorr0.5_5x162000obs_
# hshoe2det3_origfns_5k_50sims_compiled.RData       125 sims available:   nfdsmallbias_mutcorr0.5_5x165000obs_
#
# hshoe4det1_origfns_1k_50sims_compiled.RData       66 available          hshoesmallbias_mutcorr0.5_5x161000obs_
# hshoe4det1_origfns_2k_50sims_compiled.RData       50 available          hshoesmallbias_mutcorr0.5_5x162000obs_
# hshoe4det1_origfns_5k_50sims_compiled.RData       34 avilable           hshoesmallbias_mutcorr0.5_5x165000obs_


# MOD FCNS
# hshoe2det3_modfns_1k_50sims_compiled.RData        64 available:         meanfssmallbias_5x16_origmodsupint_p100_mcor.5_1000obs_

# hshoe2det3_modfns_5k_50sims_compiled.RData        52 available:         meanfssmallbias_5x16_origmodsupint_p100_mcor.5_5000obs_

# hshoe4det1


stem <- here::here("final_sims", "results", "hshoesmallbias_mutcorr0.5_5x165000obs_")
modfcns_TF <- grepl("meanfs", stem)
n_sims = 50 

if (modfcns_TF){
  reconstruct_fcn <- reconstruct_meanfcndat
  true_vec <- rep(0, 108)
  true_vec[1:8] <- 1
  fname_suffix <- "modfcns_bfdr_arr"   
} else {
  reconstruct_fcn <- reconstruct_flistdat
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


max_bfdrs <- c(0.01, 0.05, 0.1, .25)

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

cat_color("results saved to ", compiled_fname)



t(apply(ktc50k_bfdr_arr, c(2, 3), mean))
t(apply(ktc50k_bfdr_arr, c(2, 3), sd))

t(apply(ksn50k_bfdr_arr, c(2, 3), mean))
t(apply(ksn50k_bfdr_arr, c(2, 3), sd))

t(apply(ksntc50k_bfdr_arr, c(2, 3), mean))
t(apply(ksntc50k_bfdr_arr, c(2, 3), sd))


t(apply(ktc_metrictest_bfdr_arr, c(2, 3), mean))
t(apply(ktc_metrictest_bfdr_arr, c(2, 3), sd))

t(apply(ksn_metrictest_bfdr_arr, c(2, 3), mean))
t(apply(ksn_metrictest_bfdr_arr, c(2, 3), sd))

t(apply(ksntc_metrictest_bfdr_arr, c(2, 3), mean))
t(apply(ksntc_metrictest_bfdr_arr, c(2, 3), sd))







# ksn50k_bfdr0.01
# ksn50k_bfdr0.05
# ksn50k_bfdr0.1
# ksn50k_bfdr0.25
# 
# ksntc50k_bfdr0.01
# ksntc50k_bfdr0.05
# ksntc50k_bfdr0.1
# ksntc50k_bfdr0.25
# 
# apply(ksn50k_bfdr0.01, 2, mean)
# apply(ksn50k_bfdr0.01, 2, sd)
# apply(ksn50k_bfdr0.05, 2, mean)
# apply(ksn50k_bfdr0.05, 2, sd)
# apply(ksn50k_bfdr0.1, 2, mean)
# apply(ksn50k_bfdr0.1, 2, sd)
# cat_color("pause")
# apply(ksntc50k_bfdr0.01, 2, mean)
# apply(ksntc50k_bfdr0.01, 2, sd)
# apply(ksntc50k_bfdr0.05, 2, mean)
# apply(ksntc50k_bfdr0.05, 2, sd)
# apply(ksntc50k_bfdr0.1, 2, mean)
# apply(ksntc50k_bfdr0.1, 2, sd)
# cat_color("pause")
# apply(ksn_metrictest_bfdr0.01, 2, mean)
# apply(ksn_metrictest_bfdr0.01, 2, sd)
# apply(ksn_metrictest_bfdr0.05, 2, mean)
# apply(ksn_metrictest_bfdr0.05, 2, sd)
# apply(ksn_metrictest_bfdr0.1, 2, mean)
# apply(ksn_metrictest_bfdr0.1, 2, sd)
# cat_color("pause")
# apply(ksntc_metrictest_bfdr0.01, 2, mean)
# apply(ksntc_metrictest_bfdr0.01, 2, sd)
# apply(ksntc_metrictest_bfdr0.05, 2, mean)
# apply(ksntc_metrictest_bfdr0.05, 2, sd)
# apply(ksntc_metrictest_bfdr0.1, 2, mean)
# apply(ksntc_metrictest_bfdr0.1, 2, sd)
# cat_color("pause")







# 1000 obs
# sntc correction clearly superior when using last epoch kappas
# using kappas from best test_mse with only the sn correction gives nearly identical results to using sntc correction
# 5000 obs   same!










