# competitor test functions


#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
# modified forward portion of torch_horseshoe_klcorrected
source(here("Rcode", "torch_horseshoe_klcorrected.R")) 
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))

# retrieve data fcn----
reconstruct_simdat <- function(
  sim_seed,
  sim_params,
  simdat_fcn=sim_func_data
){
  # sim_seed <- 515157
  
  ## generate data ----
  set.seed(sim_seed)
  torch_manual_seed(sim_seed)
  
  simdat <- simdat_fcn(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    err_sigma = sim_params$err_sig,
    xdist = sim_params$xdist,
    xcov = sim_params$xcov,
    standardize = false_if_null(sim_params$standardize)
  )
  
  return(simdat)
}

## reconstruct data ----
stem <- here::here("sims", "results", "detlayers_532_minibatch1250obs_")
# extract sim information from first in series
first_sim <- paste0(stem, "453598.RData")
load(first_sim)
sim_seeds <- sim_res$sim_params$sim_seeds
res_fnames <- paste0(stem, sim_seeds, ".RData")

mod_fnames <- paste0(stem, sim_seeds, ".pt")
sim_params <- sim_res$sim_params
sim_params$n_obs <- 125*10

corr_fcn <- function(i, j) {0.3^(abs(i-j))}
sim_params$xcov <- make_Covmat(sim_params$d_in, corr_fcn)
sim_params$xcov



simdat <- reconstruct_simdat(
  sim_seed = sim_seeds[1],
  sim_params,
  simdat_fcn = sim_func_data
)

simdat_df <- data.frame(
  "y" = as_array(simdat$y),
  "x" = as_array(simdat$x)
)

true_inclusion = rep(FALSE, simdat$d_in)
true_inclusion[1:simdat$d_true] <- TRUE

# LM ----
lm_fit <- lm(y ~ ., data = simdat_df)
lm_pvals <- summary(lm_fit)$coef[-1, 4]
BH_pvals <- p.adjust(lm_pvals, method = "BH")
BH_decisions <- round(BH_pvals, 4) < 0.05

lm_bin_err <- binary_err_rate(est = BH_decisions, tru = true_inclusion)


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

ss_bin_err <- binary_err_rate(est = ss_median_mod, true_inclusion)
ss_bin_err


# one advantage of using neural networks over BART is that computation time for BART grows exponentially with the number of observations
# while computation time for neural networks grows only linearly with observations and quadratically with number of parameters
# so for smaller n, BART has both computational and accuracy advantages, while for very large n, neural networks have computational advantages and little if any accuracy disadvantage


# SpikeslabGAM ----
library(spikeslab)
library(spikeSlabGAM)
simdat_df

f1_string <- paste0("y ~ ", paste0("x.", 1:104, collapse = " + "))
f1 <- as.formula(f1_string) 
library(spikeSlabGAM)
options(mc.cores = 2)
Sys.time()
m <- spikeSlabGAM(formula=f1, data=simdat_df)
Sys.time()
sum <- summary(m)

print(sum, printModels=FALSE)


###################################################
### summary1.2
###################################################
#this is just ctrl-c-v from summary.spikeSlabGAM s.t. we don't get too much redundant 
# info about model formula etc. 
# roughly the same as: print(sum, printPGamma=FALSE, printModels=TRUE) 
cat("\nPosterior model probabilities (inclusion threshold =",sum$thresh,"):\n")
modelTable <- {
  #make ("x","")-vectors out of model names
  models <- sapply(names(sum$modelTable), function(x){
    sapply(strsplit(gsub("1", "x", x),""),
           function(y) gsub("0", "", y))
  })
  models <- rbind(round(sum$modelTable,3), models, round(cumsum(sum$modelTable), 3))
  rownames(models) <- c("prob.:", names(sum$predvars[!grepl("u(", names(sum$predvars), fixed=TRUE)]), "cumulative:")
  models <- data.frame(models)
  showModels <- 8
  models <- models[,1:showModels, drop=FALSE]
  colnames(models) <- 1:NCOL(models)
  models
}   
print(modelTable)

# 





# SoftBart
library(SoftBart)

num_burn <- 5000
num_save <- 5000

sbfit <- softbart(
  X = simdat_df[1:10000, -1],
  Y = simdat_df$y[1:10000],
  X_test = simdat_df[1:250 + 10000, -1]
)

# sbfit <- softbart(
#   X = CM2dat$x[1:500, ],
#   Y = CM2dat$y[1:500],
#   X_test = CM2dat$x[1:125 + 500, ]
# )

sbpost_probs <- posterior_probs(sbfit)
plot(sbpost_probs$post_probs)

print(sbpost_probs$median_probability_model)


# bartMachine
options(java.parameters = "-Xmx5g")  # enable bartMachine to use 10gb RAM
library(bartMachine)

bart_fit <- bartMachine(X = simdat_df[, -1], y = simdat_df$y)
bart_vs <- var_selection_by_permute(bart_fit)


# BART ----
library(BART)
install.packages(here::here("bart","BartMixVs_1.0.0.tar.gz"), repos = NULL, type = "source")
library(BartMixVs)

set.seed(314)
burn=10000; nd=10000

bf = wbart(
  x.train = simdat_df[1:1000, 2:105], 
  y.train = simdat_df$y[1:1000],
  ntree = 20L,
  nskip = burn,
  ndpost = nd,
  printevery = 1000)

#compute row percentages
percount = bf$varcount/apply(bf$varcount,1,sum)
# mean of row percentages
mvp =apply(percount,2,mean)
#quantiles of row percentags
qm = apply(percount,2,quantile,probs=c(.05,.95))

print(mvp)

rgy = range(qm)
p <- sim_params$d_in
plot(c(1,p),rgy,type="n",xlab="variable",ylab="post mean, percent var use",axes=FALSE)
axis(1,at=1:p,labels=names(mvp),cex.lab=0.7,cex.axis=0.7)
axis(2,cex.lab=1.2,cex.axis=1.2)
lines(1:p,mvp,col="black",lty=4,pch=4,type="b",lwd=1.5)
for(i in 1:p) {
  lines(c(i,i),qm[,i],col="blue",lty=3,lwd=1.0)
}















# scenario CM2
# n= 500 or 1000
# p = 84
# sig^2 = 1
# 
# 
# x_1 through x_20 ~ Bernoulli (0.2)
# x_21 though x_40 ~ Bernoulli (0.5)
# x_41 though x_84 ~ MVN(0, 1, corr = 0.3)
# y ~ Normal(f(x), sig^2), where 
friedman_fx <- function(x) {
  -4 + x[, 1] +
    sin(pi*x[, 1]*x[, 44]) -
    x[, 21] +
    0.6*x[, 41]*x[, 42] -
    exp( -2*(x[, 42]+1)^2 ) -
    x[, 43]^2 +
    0.5*x[, 44]
}
friedman_true_vars <- c(1, 21, 41:44)

gen_simdat_CM2 <- function(
    n_obs = 625,
    p = 84,
    sig = 1,
    mvncorr = 0.3,
    fx = friedman_fx
){
  
  x1to20 <- matrix(
    rbinom(20 * n_obs, 1, 0.2),
    ncol = 20
  )
  x21to40 <- matrix(
    rbinom(20 * n_obs, 1, 0.5),
    ncol = 20
  )
  xcov <- diag(44)
  for (i in 1:nrow(xcov)){
    if (i < nrow(xcov)){
      xcov[i, i+1] <- mvncorr      
    }
    if (i > 1){
      xcov[i, i-1] <- mvncorr      
    }
  }
  x41to84 <- MASS::mvrnorm(n=n_obs, mu=rep(0, 44), Sigma = xcov)
  X <- cbind(x1to20, x21to40, x41to84)
  
  Ey <- fx(X)
  y <- Ey + rnorm(length(Ey), 0, sd = sig)
  
  
  
  return(
    list(
      "x" = X,
      "y" = y,
      "Ey" = Ey
    )
  )
}

CM2dat <- gen_simdat_CM2()



# precision = TP / TP + FP   = 1 - FDR
# recall = TP / TP + FN
# F1 = 2*precision*recall / precision+recall
# r_mis = # times recall < 1







