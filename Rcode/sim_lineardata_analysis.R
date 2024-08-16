##################################################
## Project:   sim_lineardata.R analysis
## Date:      Aug 13, 2024
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
testing <- FALSE

fname <- ifelse(testing, "linear_sim_sig1_TEST.Rdata", "linear_sim_sig1.Rdata")
fpath <- here("Rcode", "results", fname)

load(fpath)

names(res[[1]])

# replace with true coefs
# tru <- true_coefs!=0
tru <-c(rep(1, 4), rep(0, 100))


# slnj bare results

# threshold log_alphas at -3 (exp(-3) \approx 0.05)
# what is a reasonable dropout probability?  
log_alpha_threshold <- -3
txt <- paste0("log_alpha_threshold = ", log_alpha_threshold, 
              " (i.e. dropout rate of <",
              round(exp(log_alpha_threshold), 5), 
              " required for inclusion)")
cat_color(txt)

slnj_log_alphas <- t(sapply(res, function(X) X$slnj$log_dropout_alphas))
slnj_include <- slnj_log_alphas < log_alpha_threshold

slnj_binary_err <- t(apply(
  slnj_include, 1, 
  function(X) binary_err(est = X, tru = tru)
))
slnj_binary_err_rates <- t(apply(
  slnj_include, 1,
  function(X) binary_err_rate(est = X, tru = tru)
))

slnj_binary_err_rates
slnj_fwer <- mean(slnj_binary_err_rates[,1] != 0)

# other slnj metrics
slnj_metrics <- t(sapply(res, function(X) X$slnj$other_metrics))








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

slnj_binary_err_rates
ss_binary_err_rates
lasso_binary_err_rates


