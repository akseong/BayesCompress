# competitor test functions


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

# retrieve data fcn----
reconstruct_meanfcndat <- function(
    sim_seed,
    sim_params
){
  # sim_seed <- 515157
  
  ## generate data ----
  set.seed(sim_seed)
  torch_manual_seed(sim_seed)
  
  simdat <- sim_meanfcn_data(
    n_obs = sim_params$n_obs, 
    d_in = sim_params$d_in, 
    genXfcn = sim_params$genXfcn,
    meanfcn = sim_params$meanfcn,
    err_sigma = sim_params$err_sig, 
    round_dig = sim_params$round_dig, 
    standardize = FALSE
  )
  
  return(simdat)
}



reconstruct_flistdat <- function(
    sim_seed,
    sim_params
){
  # sim_seed <- 515157
  
  ## generate data ----
  set.seed(sim_seed)
  torch_manual_seed(sim_seed)
  
  simdat <- sim_func_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    err_sigma = sim_params$err_sig,
    xdist = sim_params$xdist,
    xcov = sim_params$xcov,
    mut_corr = sim_params$mut_corr,
    standardize = false_if_null(sim_params$standardize)
  )
  
  return(simdat)
}









## reconstruct data ----
stem <- here::here("sims", "results", "nfdsmallbias_mutcorr.5_5x162000obs_155447")
# extract sim information from first in series
first_sim <- paste0(stem, ".RData")
load(first_sim)
sim_seeds <- sim_res$sim_params$sim_seeds
res_fnames <- paste0(stem, sim_seeds, ".RData")

mod_fnames <- paste0(stem, sim_seeds, ".pt")
sim_params <- sim_res$sim_params


meanfcn <- function(x, round_dig = NULL){
  2*(sin(pi*x[, 1]))*(x[,2]*(x[, 2]>0)) - 2*(x[, 2]<0) + 
    x[,3]/(1 + x[,4]*(x[, 5]>0))
}

meanfcn2 <- function(x, round_dig = NULL){
  (sin(pi*x[, 1]/2))*(x[, 2]>0) - (x[, 2]<0) + x[,1]*(x[,1] < 0) + 
    x[,3]/(1 + x[,4] + x[, 5]*(x[, 5]>0))
}

meanfcn_origmod <- function(x, round_dig = NULL){
  - cos(pi/1.5*x[, 1])*(x[,1]>0) - (x[,1]<0) + 
    cos(pi/2*x[,2])*(x[,2]<0) + sin(pi/1.5*x[,2])*(x[,2]>0) - 
    x[, 3]/(1 + x[,4]^2) + 1 / (1 + 2*x[,5]*(x[,5]>0))
}


meanfcn_origmod_extra <- function(x, round_dig = NULL){
  -cos(pi/1.5*x[, 1])*(x[,1]>0) - (x[,1]<0) + 
    cos(pi/2*x[,2])*(x[,2]<0) + sin(pi/1.5*x[,2])*(x[,2]>0) - 
    x[, 3]/(1 + x[,4]^2) + 1 / (1 + 2*x[,5]*(x[,5]>0)) - 
    x[,6]^2/4 + abs(x[,7])^.75 - sin(pi/1.2*x[,8]) + cos(pi*x[,8])
}

orig_fcns <- function(x, round_dig = NULL){
  -cos(pi/1.5*x[,1]) + cos(pi*x[,2]) + sin(pi/1.2*x[,2]) + 
    abs(x[,3])^(.75) - x[,4]^2/4
  }
# meanfcn_Liang1.5 <- function(X, round_dig = NULL){
#   Ey <- X[, 2] / (1 + X[, 1]^2) + sin(X[, 3]*X[, 4]) + X[, 5]*(2*(X[,5]>0) - 1*(X[,5]<0))
#   if (!is.null(round_dig)) {Ey <- round(Ey, round_dig)}
#   return(Ey)
# }
orig_fcns_limsup <- function(x, round_dig = NULL){
  -cos(pi/1.5*x[,1])*(x[,1]<0) + cos(pi*x[,2])*(x[,2] > 0) + sin(pi/1.2*x[,2])*(x[,2]<0) + 
    abs(x[,3])^(.75) - x[,4]^2/4
}


sim_params$n_obs <- 2000  # at 500 & 1000, gets x1, x2, x3 typically; at 2000 sometimes gets x4;  at 5000 gets all 4
sim_params$d_in <- 100
sim_params$meanfcn <- orig_fcns
sim_params$mut_corr <- 0.5
n_sims = 3
seeds <- round(runif(n_sims, 0, 10000))

res <- list(
  "seeds" = seeds
)

# sim_params$mut_corr
# sim_params$standardize <- FALSE
# simdat <- reconstruct_flistdat(sim_seed = sim_params$sim_seeds[1], sim_params)




lm_errs <- matrix(NA, ncol = n_sims, nrow = 4)
rownames(lm_errs) <- c("FP", "TP", "FN", "TN")
ss_sums <- sb_posts <- matrix(NA, ncol = n_sims, nrow = sim_params$d_in)

for (s_i in 1:n_sims){
print(s_i)
set.seed(seeds[s_i])
simdat <- sim_meanfcn_data(
  n_obs = sim_params$n_obs,
  d_in = sim_params$d_in,
  mut_corr = sim_params$mut_corr,
  genXfcn = genX_mutualcorr,
  meanfcn = sim_params$meanfcn,
  err_sigma = sim_params$err_sig,
  round_dig = sim_params$round_dig,
  standardize = FALSE
)

simdat_df_raw <- data.frame(
  "y" = as_array(simdat$y),
  "x" = as_array(simdat$x)
)

n_ttsplit <- sim_params$ttsplit * sim_params$n_obs
simdat_tr <- simdat_df_raw[1:sim_params$n_obs, ]
simdat_test <- simdat_df_raw[1:(sim_params$n_obs - n_ttsplit), ]

# standardize
scale_list <- scale_mat(simdat_tr)
simdat_df <- scale_list$scaled
simdat_df_test <- scale_mat(simdat_test, means = scale_list$means, sds = scale_list$sds)$scaled

true_inclusion = rep(FALSE, simdat$d_in)
true_inclusion[1:4] <- TRUE

# LM ----
lm_fit <- lm(y ~ ., data = simdat_df)
lm_pvals <- summary(lm_fit)$coef[-1, 4]
BH_pvals <- p.adjust(lm_pvals, method = "BH")
BH_decisions <- round(BH_pvals, 4) < 0.05

lm_bin_err <- binary_err_rate(est = BH_decisions, tru = true_inclusion)
lm_bin_err

lm_errs[, s_i] <- lm_bin_err

# Spike-slab ----
library(BoomSpikeSlab)
modmat <- cbind(1, simdat_df[, -1])
prior = IndependentSpikeSlabPrior(modmat, simdat_df$y, 
                                  expected.model.size = 20,
                                  prior.beta.sd = rep(1, ncol(modmat))) 

ss_fit = lm.spike(y ~ ., data = simdat_df, niter = 1000, prior = prior, ping = 0)
ss_summ <- summary(ss_fit)$coef

## sort spike-slab results to appear in same order as data
ss_summ_rnames <- rownames(ss_summ)
var_names <- names(simdat_df) # list variables in order appearing in data
var_names[1] <- "(Intercept)" # replace "y" with intercept
ss_summ_order <- match(var_names, ss_summ_rnames)
ss_summ_sorted <- ss_summ[ss_summ_order, ]

# get median model based on PIPs, ignore intercept
ss_median_mod <- ss_summ_sorted[-1, 5] > 0.5 
ss_median_mod

# err_by_max_bfdr(1 - ss_summ_sorted[-1, "inc.prob"], true_vec = true_inclusion)

ss_bin_err <- binary_err_rate(est = ss_median_mod, true_inclusion)
ss_bin_err

ss_sums[, s_i]  <- ss_summ_sorted[-1, 5]

# one advantage of using neural networks over BART is that computation time for BART grows exponentially with the number of observations
# while computation time for neural networks grows only linearly with observations and quadratically with number of parameters
# so for smaller n, BART has both computational and accuracy advantages, while for very large n, neural networks have computational advantages and little if any accuracy disadvantage


# # SpikeslabGAM ----
# library(spikeslab)
# library(spikeSlabGAM)
# 
# f1_string <- paste0("y ~ ", paste0("x.", 1:sim_params$d_in, collapse = " + "))
# f1 <- as.formula(f1_string)
# library(spikeSlabGAM)
# options(mc.cores = 2)
# Sys.time()
# m <- spikeSlabGAM(formula=f1, data=simdat_df)
# Sys.time()
# sum <- summary(m)
# 
# print(sum, printModels=FALSE)
# 
# 
# ###################################################
# ### summary1.2
# ###################################################
# #this is just ctrl-c-v from summary.spikeSlabGAM s.t. we don't get too much redundant
# # info about model formula etc.
# # roughly the same as: print(sum, printPGamma=FALSE, printModels=TRUE)
# cat("\nPosterior model probabilities (inclusion threshold =",sum$thresh,"):\n")
# modelTable <- {
#   #make ("x","")-vectors out of model names
#   models <- sapply(names(sum$modelTable), function(x){
#     sapply(strsplit(gsub("1", "x", x),""),
#            function(y) gsub("0", "", y))
#   })
#   models <- rbind(round(sum$modelTable,3), models, round(cumsum(sum$modelTable), 3))
#   rownames(models) <- c("prob.:", names(sum$predvars[!grepl("u(", names(sum$predvars), fixed=TRUE)]), "cumulative:")
#   models <- data.frame(models)
#   showModels <- 8
#   models <- models[,1:showModels, drop=FALSE]
#   colnames(models) <- 1:NCOL(models)
#   models
# }
# print(modelTable)







# SoftBart
library(SoftBart)

sbfit <- softbart(
  X = simdat_df[, -1],
  Y = simdat_df[, 1],
  X_test = simdat_df_test[, -1]
)

# sbfit <- softbart(
#   X = CM2dat$x[1:500, ],
#   Y = CM2dat$y[1:500],
#   X_test = CM2dat$x[1:125 + 500, ]
# )

sbpost_probs <- posterior_probs(sbfit)
plot(sbpost_probs$post_probs)

print(sbpost_probs$median_probability_model)

sb_posts[, s_i] <- sbpost_probs$post_probs
print(sim_params$n_obs)
print(sim_params$mut_corr)
}

lm_errs
ss_sums
sb_posts
print(seeds)
print(sim_params$n_obs)
print(sim_params$mut_corr)
print(sim_params$meanfcn)

# SOFTBART, p = 20
# orig_fcns <- function(x, round_dig = NULL){
#   -cos(pi/1.5*x[,1]) + cos(pi*x[,2]) + sin(pi/1.2*x[,2])
#   + abs(x[,3])^(.75) - x[,4]^2/4
# }
# mcor 0:
# - 1000 obs:
# - 2000 obs:
# - 5000 obs: x3, x4, no FPs

# mor 0.25:
# - 1000 obs: x3, x4, maybe 1 FP (0.81 PIP)
# - 2000 obs: x3, x4, no FPs
# - 5000 obs: x3, x4, no FPS

# mcor 0.5
# - 1000 obs: x3, sometimes x4
# - 2000 obs: x3, x4, no FPs
# - 5000 obs: x3, x4, no FPs


# function(x, round_dig = NULL){
#   -cos(pi/1.5*x[, 1])*(x[,1]>0) - (x[,1]<0)
#   + cos(pi/2*x[,2])*(x[,2]<0) + sin(pi/1.5*x[,2])*(x[,2]>0)
#   - x[, 3]/(1 + x[,4]^2) + 1 / (1 + 2*x[,5]*(x[,5]>0))
# }
# mcor 0:
# - 1000 obs:
# - 2000 obs:
# - 5000 obs:

# mor 0.25:
# - 1000 obs: x3, x5; x4 around .55 PIP max
# - 2000 obs:
# - 5000 obs:

# mcor 0.5
# - 1000 obs: x3, x5; x4 around .85 PIP
# - 2000 obs: x3, x5, mostly x4
# - 5000 obs:















# 
# # bartMachine
# options(java.parameters = "-Xmx5g")  # enable bartMachine to use 10gb RAM
# library(bartMachine)
# 
# bart_fit <- bartMachine(X = simdat_df[, -1], y = simdat_df$y)
# bart_vs <- var_selection_by_permute(bart_fit)
# 
# 
# # BART ----
# library(BART)
# install.packages(here::here("bart","BartMixVs_1.0.0.tar.gz"), repos = NULL, type = "source")
# library(BartMixVs)
# 
# set.seed(314)
# burn=10000; nd=10000
# 
# bf = wbart(
#   x.train = simdat_df[1:1000, 2:105], 
#   y.train = simdat_df$y[1:1000],
#   ntree = 20L,
#   nskip = burn,
#   ndpost = nd,
#   printevery = 1000)
# 
# #compute row percentages
# percount = bf$varcount/apply(bf$varcount,1,sum)
# # mean of row percentages
# mvp =apply(percount,2,mean)
# #quantiles of row percentags
# qm = apply(percount,2,quantile,probs=c(.05,.95))
# 
# print(mvp)
# 
# rgy = range(qm)
# p <- sim_params$d_in
# plot(c(1,p),rgy,type="n",xlab="variable",ylab="post mean, percent var use",axes=FALSE)
# axis(1,at=1:p,labels=names(mvp),cex.lab=0.7,cex.axis=0.7)
# axis(2,cex.lab=1.2,cex.axis=1.2)
# lines(1:p,mvp,col="black",lty=4,pch=4,type="b",lwd=1.5)
# for(i in 1:p) {
#   lines(c(i,i),qm[,i],col="blue",lty=3,lwd=1.0)
# }
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # scenario CM2
# # n= 500 or 1000
# # p = 84
# # sig^2 = 1
# # 
# # 
# # x_1 through x_20 ~ Bernoulli (0.2)
# # x_21 though x_40 ~ Bernoulli (0.5)
# # x_41 though x_84 ~ MVN(0, 1, corr = 0.3)
# # y ~ Normal(f(x), sig^2), where 
# friedman_fx <- function(x) {
#   -4 + x[, 1] +
#     sin(pi*x[, 1]*x[, 44]) -
#     x[, 21] +
#     0.6*x[, 41]*x[, 42] -
#     exp( -2*(x[, 42]+1)^2 ) -
#     x[, 43]^2 +
#     0.5*x[, 44]
# }
# friedman_true_vars <- c(1, 21, 41:44)
# 
# gen_simdat_CM2 <- function(
#     n_obs = 625,
#     p = 84,
#     sig = 1,
#     mvncorr = 0.3,
#     fx = friedman_fx
# ){
#   
#   x1to20 <- matrix(
#     rbinom(20 * n_obs, 1, 0.2),
#     ncol = 20
#   )
#   x21to40 <- matrix(
#     rbinom(20 * n_obs, 1, 0.5),
#     ncol = 20
#   )
#   xcov <- diag(44)
#   for (i in 1:nrow(xcov)){
#     if (i < nrow(xcov)){
#       xcov[i, i+1] <- mvncorr      
#     }
#     if (i > 1){
#       xcov[i, i-1] <- mvncorr      
#     }
#   }
#   x41to84 <- MASS::mvrnorm(n=n_obs, mu=rep(0, 44), Sigma = xcov)
#   X <- cbind(x1to20, x21to40, x41to84)
#   
#   Ey <- fx(X)
#   y <- Ey + rnorm(length(Ey), 0, sd = sig)
#   
#   
#   
#   return(
#     list(
#       "x" = X,
#       "y" = y,
#       "Ey" = Ey
#     )
#   )
# }
# 
# CM2dat <- gen_simdat_CM2()



# precision = TP / TP + FP   = 1 - FDR
# recall = TP / TP + FN
# F1 = 2*precision*recall / precision+recall
# r_mis = # times recall < 1







