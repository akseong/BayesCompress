





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


# n=1000, mcorr = 0.5, 5x16 2 hshoe 3 deterministic layers ----

stem <- here::here("final_sims", "results", "nfdsmallbias_mutcorr0.5_5x165000obs_")
true_vec <- rep(0, 104)
true_vec[1:4] <- 1
# started off some with 5, some with 10.  Figure out which ones have 5, vs 10
overall_seeds <- c(516, as.numeric(paste0(516, 0:13)))
possible_sim_seeds <- c()

for (i in 1:length(overall_seeds)){
  set.seed(overall_seeds[i])
  possible_sim_seeds <- c(possible_sim_seeds, floor(runif(n = 10, 0, 1000000)))
}

# which of these files exist
poss_fnames <- paste0(stem, possible_sim_seeds, ".RData")
exists_TF <- 
  has_corrections_by_layer <- rep(F, length(poss_fnames))

for (i in 1:length(poss_fnames)){
  exists_TF[i] <- file.exists(poss_fnames[i])
}

sim_fnames <- paste0(stem, possible_sim_seeds[exists_TF], ".RData")
mod_fnames <- paste0(stem, possible_sim_seeds[exists_TF], ".pt")

ksn50k_mat <- 
  ksntc50k_mat <- 
  ksn_metrictest_mat <- 
  ksntc_metrictest_mat <-   matrix(NA, nrow = length(sim_fnames), ncol = 104)

# quick check of sn and sntc corrected kappas at last epoch
metrictest_rows <- rep(NA, length(sim_fnames))
for (f_ind in 1:length(sim_fnames)){
  load(sim_fnames[f_ind])
  last_epoch <- nrow(sim_res$kappa_sn_mat)
  ksn50k_mat[f_ind, ] <- sim_res$kappa_sn_mat[last_epoch,]
  ksntc50k_mat[f_ind, ] <- sim_res$kappa_sntc_mat[last_epoch,]
  
  row_ind <- get_smallest_testmse_epoch(sim_res$loss_mat)
  metrictest_rows[f_ind] <- row_ind
  ksn_metrictest_mat[f_ind, ] <- sim_res$kappa_sn_mat[row_ind,]
  ksntc_metrictest_mat[f_ind, ] <- sim_res$kappa_sntc_mat[row_ind,]
}


sum(ksn50k_mat[, 1:4] < 0.05)     # 1000obs: 126 out of 300    # 5000obs: 206 
sum(ksn50k_mat[, 5:104] < 0.5)    # 1000obs: no FPs            # 5000obs: 0

sum(ksntc50k_mat[, 1:4] < 0.05)   # 1000obs: 191 out of 300    # 5000obs: 351
sum(ksntc50k_mat[, 5:104] < 0.5)  # 1000obs: no FPs            # 5000obs: 100


metrictest_rows
sim_res$loss_mat[metrictest_rows,5] # KLweight at best test_mse
sum(ksn_metrictest_mat[, 1:4] < 0.05) # 17 out of 300          # 5000obs: 25
sum(ksn_metrictest_mat[, 5:104] < 0.5)  # no FPs               # 5000obs: 0

sum(ksntc_metrictest_mat[, 1:4] < 0.05) # 156 out of 300       # 5000obs: 313
sum(ksntc_metrictest_mat[, 5:104] < 0.5)  # 8                  # 5000obs: 11




max_bfdrs <- c(0.01, 0.05, 0.1, .25)

ksn50k_bfdr0.01 <- matrix(NA, nrow = nrow(ksn50k_mat), ncol = 8)
colnames(ksn50k_bfdr0.01) <- c("max_bfdr", "fdr", "bfdr", "FPR", "TPR_sens_recall", "FNR", "TN_specificity", "f1")

ksn50k_bfdr0.05 <- 
  ksn50k_bfdr0.1 <- 
  ksn50k_bfdr0.25 <- 
  ksntc50k_bfdr0.01 <- 
  ksntc50k_bfdr0.05 <- 
  ksntc50k_bfdr0.1 <- 
  ksntc50k_bfdr0.25 <- 
  ksn_metrictest_bfdr0.01 <- 
  ksn_metrictest_bfdr0.05 <- 
  ksn_metrictest_bfdr0.1 <- 
  ksn_metrictest_bfdr0.25 <-
  ksntc_metrictest_bfdr0.01 <- 
  ksntc_metrictest_bfdr0.05 <- 
  ksntc_metrictest_bfdr0.1 <- 
  ksntc_metrictest_bfdr0.25 <-ksn50k_bfdr0.01

for (k in 1:nrow(ksn50k_mat)){
  err_mat <- metrics_err_by_max_bfdr(dropout_vec = ksn50k_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ksn50k_bfdr0.01[k, ] <- c(err_mat[1, ])
  ksn50k_bfdr0.05[k, ] <- c(err_mat[2, ])
  ksn50k_bfdr0.1[k, ] <- c(err_mat[3, ])
  ksn50k_bfdr0.25[k, ] <- c(err_mat[4, ])
  
  err_mat_sntc <- metrics_err_by_max_bfdr(dropout_vec = ksntc50k_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ksntc50k_bfdr0.01[k, ] <- c(err_mat_sntc[1, ])
  ksntc50k_bfdr0.05[k, ] <- c(err_mat_sntc[2, ])
  ksntc50k_bfdr0.1[k, ] <- c(err_mat_sntc[3, ])
  ksntc50k_bfdr0.25[k, ] <- c(err_mat_sntc[4, ])
  
  err_mat_metrictest <- metrics_err_by_max_bfdr(dropout_vec = ksn_metrictest_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ksn_metrictest_bfdr0.01[k, ] <- c(err_mat_sntc[1, ])
  ksn_metrictest_bfdr0.05[k, ] <- c(err_mat_sntc[2, ])
  ksn_metrictest_bfdr0.1[k, ] <- c(err_mat_sntc[3, ])
  ksn_metrictest_bfdr0.25[k, ] <- c(err_mat_sntc[4, ])
  
  err_mat_metrictest <- metrics_err_by_max_bfdr(dropout_vec = ksntc_metrictest_mat[k, ], true_vec, bfdr_vec = max_bfdrs)
  ksntc_metrictest_bfdr0.01[k, ] <- c(err_mat_sntc[1, ])
  ksntc_metrictest_bfdr0.05[k, ] <- c(err_mat_sntc[2, ])
  ksntc_metrictest_bfdr0.1[k, ] <- c(err_mat_sntc[3, ])
  ksntc_metrictest_bfdr0.25[k, ] <- c(err_mat_sntc[4, ])
}


ksn50k_bfdr0.01
ksn50k_bfdr0.05
ksn50k_bfdr0.1
ksn50k_bfdr0.25

ksntc50k_bfdr0.01
ksntc50k_bfdr0.05
ksntc50k_bfdr0.1
ksntc50k_bfdr0.25

apply(ksn50k_bfdr0.01, 2, mean)
apply(ksn50k_bfdr0.01, 2, sd)
apply(ksn50k_bfdr0.05, 2, mean)
apply(ksn50k_bfdr0.05, 2, sd)
apply(ksn50k_bfdr0.1, 2, mean)
apply(ksn50k_bfdr0.1, 2, sd)
cat_color("pause")
apply(ksntc50k_bfdr0.01, 2, mean)
apply(ksntc50k_bfdr0.01, 2, sd)
apply(ksntc50k_bfdr0.05, 2, mean)
apply(ksntc50k_bfdr0.05, 2, sd)
apply(ksntc50k_bfdr0.1, 2, mean)
apply(ksntc50k_bfdr0.1, 2, sd)
cat_color("pause")
apply(ksn_metrictest_bfdr0.01, 2, mean)
apply(ksn_metrictest_bfdr0.01, 2, sd)
apply(ksn_metrictest_bfdr0.05, 2, mean)
apply(ksn_metrictest_bfdr0.05, 2, sd)
apply(ksn_metrictest_bfdr0.1, 2, mean)
apply(ksn_metrictest_bfdr0.1, 2, sd)
cat_color("pause")
apply(ksntc_metrictest_bfdr0.01, 2, mean)
apply(ksntc_metrictest_bfdr0.01, 2, sd)
apply(ksntc_metrictest_bfdr0.05, 2, mean)
apply(ksntc_metrictest_bfdr0.05, 2, sd)
apply(ksntc_metrictest_bfdr0.1, 2, mean)
apply(ksntc_metrictest_bfdr0.1, 2, sd)
cat_color("pause")


# 1000 obs
# sntc correction clearly superior when using last epoch kappas
# using kappas from best test_mse with only the sn correction gives nearly identical results to using sntc correction
# 5000 obs   same!










