##################################################
## Project:   MLP functional data sim analysis
## Date:      Aug 20, 2024
## Author:    Arnie Seong
##################################################

library(here)
library(tidyr)
library(dplyr)
library(parallel)

# BayesCompress
library(torch)
source(here("Rcode", "BayesianLayers.R"))
source(here("Rcode", "sim_functions.R"))


# files
fname <- "MLP_fcnl_sim_sig1"
partial_fpaths <- here("Rcode", "results", paste0(fname, "_PARTIAL", 1:7, ".Rdata"))


fname_part2 <- "MLP_fcnl_sim_sig1_part2"
partial_fpaths_part2 <- here("Rcode", "results", paste0(fname_part2, "_PARTIAL", 1:5, ".Rdata"))

partial_fpaths_all <- c(partial_fpaths, partial_fpaths_part2)

mlnj_res <- list()

for (i in 1:length(partial_fpaths_all)){
  load(file = partial_fpaths_all[i])
  mlnj_res <- append(mlnj_res, partial_res)
}


tru <- c(rep(1, 4), rep(0, 96))


# mlnj bare results

# threshold log_alphas at -3 (exp(-3) \approx 0.05)
# what is a reasonable dropout probability?  
log_alpha_threshold <- -3
txt <- paste0("log_alpha_threshold = ", log_alpha_threshold, 
              " (i.e. dropout rate of <",
              round(exp(log_alpha_threshold), 5), 
              " required for inclusion)")
cat_color(txt)

mlnj_log_alphas <- t(sapply(mlnj_res, function(X) X$mlnj$log_dropout_alphas))
mlnj_include <- mlnj_log_alphas < log_alpha_threshold


mlnj_binary_err <- t(apply(
  mlnj_include, 1, 
  function(X) binary_err(est = X, tru = tru)
))
mlnj_binary_err_rates <- t(apply(
  mlnj_include, 1,
  function(X) binary_err_rate(est = X, tru = tru)
))

mlnj_binary_err_rates
mlnj_fwer <- mean(mlnj_binary_err_rates[,1] != 0)

# other mlnj metrics
mlnj_metrics <- t(sapply(mlnj_res, function(X) X$mlnj$other_metrics))





# # # # # # # # # # # # # # # # # # # # # # # # #
## LASSO ----
# # # # # # # # # # # # # # # # # # # # # # # # #

lasso_coefs <- t(sapply(mlnj_res, function(X) as.vector(X$lasso$coefs)[-1]))
lasso_include <- lasso_coefs != 0 
lasso_binary_err_rates <- t(apply(
  lasso_include, 1, 
  function(X) binary_err_rate(est = X, tru = tru)
))

lasso_fwer <- mean(lasso_binary_err_rates[,1] != 0)


# # # # # # # # # # # # # # # # # # # # # # # # #
## Spike-&-Slab ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# use same threshold?


ss_post_inclusion_probs <- t(sapply(mlnj_res, function(X) X$ss$coefs[, 5]))
ss_dropout_equiv <- 1 - ss_post_inclusion_probs
ss_include <- ss_dropout_equiv < exp(log_alpha_threshold)


ss_binary_err_rates <- t(apply(
  ss_include, 1, 
  function(X) binary_err_rate(est = X, tru = tru)
))

ss_fwer <- mean(ss_binary_err_rates[,1] != 0)

mlnj_binary_err_rates
ss_binary_err_rates
lasso_binary_err_rates







# # # # # # # # # # # # # # # # # # # # # # # # #
## troubleshooting ----
# # # # # # # # # # # # # # # # # # # # # # # # #
# 57, 70, 85 best results

stop_reasons <- t(sapply(mlnj_res, function(X) X$mlnj$stop_reason))
stop_epoch <- sapply(mlnj_res, function(X) X$mlnj$epoch)

stop_reasons  # most stop b/c met convergence crit.  A few stop b/c max epochs reached
stop_epoch


# check loss -- mse, kl
metrics <- matrix(NA, nrow = length(mlnj_res), ncol = 3)
colnames(metrics) <- c("epoch", "mse", "kl")

for (i in 1:length(mlnj_res)){ 
 mat_row_ind <- which(mlnj_res[[i]]$mlnj$log_alpha_mat[, 101] == stop_epoch[i]) 
 metrics[i,] <- mlnj_res[[i]]$mlnj$log_alpha_mat[mat_row_ind, 101:103]
}


metrics[c(57, 70, 85),]  # best results have fairly high MSE, lowish KL
hist(metrics[, 2], breaks = 30, main = "\'best\' mses across simulations") # mses
hist(metrics[, 3], breaks = 30, main = "\'best\' kls across simulations") # kls
# suggests may want to weigh KL higher



# just looking at distribution of best MSEs, KLs found 
# (i.e. from "best" 100 training epochs for num_samps simulated datasets)
num_samps <- 10
sample_ind <- sample(x = 1:length(mlnj_res), size = num_samps)

mses <- sapply(mlnj_res[sample_ind], function(X) X$mlnj$log_alpha_mat[, 102])
kls <- sapply(mlnj_res[sample_ind], function(X) X$mlnj$log_alpha_mat[, 103])

hist(mses, breaks = 30, main = paste0("last 100 epochs' MSEs from ", num_samps, 
" trials"))
hist(kls, breaks = 100, main = paste0("last 100 epochs' KLs from ", num_samps, 
                                      " trials"))
# convergence criterion reached too early, it appears.
# KLs are very tight







# compare to SLP linear data simulation
load(file = here::here("Rcode", "results", "linear_sim_sig1.Rdata"))
slnj_res <- res
slnj_sample_ind <- sample(x = 1:length(slnj_res), size = num_samps)

slnj_mses <- sapply(slnj_res[slnj_sample_ind], function(X) X$slnj$log_alpha_mat[, 106])
slnj_kls <- sapply(slnj_res[slnj_sample_ind], function(X) X$slnj$log_alpha_mat[, 107])


hist(slnj_mses, breaks = 30, main = paste0("last 100 epochs' MSEs from ", num_samps, 
                                           " trials (slnj)"))
hist(slnj_kls, breaks = 100, main = paste0("last 100 epochs' KLs from ", num_samps, 
                                           " trials (slnj)"))





t(apply(slnj_mses, 2, summary))  # SL mses
t(apply(mses, 2, summary))  # MLNJ mses

t(apply(slnj_kls, 2, summary))  # SL KLs
t(apply(kls, 2, summary))  # MLNJ kls




# MLNJ MSEs tend to be larger than error variane (prob just b/c fcnl data)
# SLNJ MSEs tend to be smaller







