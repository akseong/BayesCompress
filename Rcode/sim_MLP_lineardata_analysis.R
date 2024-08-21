##################################################
## Project:   sim_MLP_lineardata.R and sim_MLP_lineardata_par.R analysis
## Date:      Aug 21, 2024
## Author:    Arnie Seong
##################################################


library(here)
library(tidyr)
library(dplyr)

# BayesCompress
library(torch)
# source(here("Rcode", "BayesianLayers.R"))
source(here("Rcode", "sim_functions.R"))
# 
# #competitors
# library(BoomSpikeSlab)
# library(glmnet)


# sim params --------------------------------------------------------------

fname <- "MLP_linear_sim_sig1.Rdata"
fname2 <- "MLP_linear_sim_sig1_part2.Rdata"
fpath <- here::here("Rcode", "results", fname)
fpath2 <- here::here("Rcode", "results", fname2)

load(fpath)
load(fpath2)

# some results null (stopped simulation b/c needed to use computer)
res_non_null <- res[!sapply(res, is.null)]
result_non_null <- result[!sapply(result, is.null)]

res <- append(res_non_null, result_non_null)

# replace with true coefs
tru <-c(rep(1, 4), rep(0, 100))


# mlnj bare results

# threshold log_alphas at -3 (exp(-3) \approx 0.05)
# what is a reasonable dropout probability?  
log_alpha_threshold <- -3
txt <- paste0("log_alpha_threshold = ", log_alpha_threshold, 
              " (i.e. dropout rate of <",
              round(exp(log_alpha_threshold), 5), 
              " required for inclusion)")
cat_color(txt)

mlnj_log_alphas <- t(sapply(res, function(X) X$mlnj$log_dropout_alphas))
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
mlnj_metrics <- t(sapply(res, function(X) X$mlnj$other_metrics))








# # # # # # # # # # # # # # # # # # # # # # # # #
## LASSO ----
# # # # # # # # # # # # # # # # # # # # # # # # #

lasso_coefs <- t(sapply(res, function(X) as.vector(X$lasso$coefs)[-1]))
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


ss_post_inclusion_probs <- t(sapply(res, function(X) X$ss$coefs[, 5]))
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


mlnj_res <- colMeans(mlnj_binary_err_rates)
ss_res <- colMeans(ss_binary_err_rates)
lasso_res <- colMeans(lasso_binary_err_rates)


mlnj_fwer <- mean(mlnj_binary_err_rates[,1] != 0)
ss_fwer <- mean(ss_binary_err_rates[,1] != 0)
lasso_fwer <- mean(lasso_binary_err_rates[,1] != 0)

mlnj_res <- c(mlnj_res, "FWER" = mlnj_fwer)
ss_res <- c(ss_res, "FWER" = ss_fwer)
lasso_res <- c(lasso_res, "FWER" = lasso_fwer)

mlnj_res
ss_res
lasso_res

