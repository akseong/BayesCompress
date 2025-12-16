# competitor test functions


#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
# modified forward portion of torch_horseshoe_klcorrected
source(here("Rcode", "torch_horseshoe_scaledacts.R")) 
source(here("Rcode", "sim_functions.R"))
# source(here("Rcode", "sim_hshoe_normedresponse.R"))


# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) - (abs(x))
# flist = list(fcn1, fcn2, fcn3, fcn4)
# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) -x^2 / 2 -3
# fcn4 <- function(x) - log(abs(x) + 1e-3)
fcn1 <- function(x) -cos(pi/1.5*x)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
fcn3 <- function(x) abs(x)^(.75)
fcn4 <- function(x) -x^2 / 4
flist = list(fcn1, fcn2, fcn3, fcn4)
plot_datagen_fcns(flist)


save_mod_path_prestem <- here::here(
  "sims", 
  "results", 
  "hshoe_smooth_kaiming3232_"
)

sim_params <- list(
  "sim_name" = "tau_0 = 1, kaiming init, 2 layers 32 32, nobatching, fcnal data.  ",
  "seed" = 23232,
  "n_sims" = 1, 
  "train_epochs" = 5E5,
  "report_every" = 1E4,
  "use_cuda" = use_cuda,
  "d_in" = 104,
  "d_hidden1" = 32,
  "d_hidden2" = 32,
  # "d_hidden3" = 8,
  "d_out" = 1,
  "n_obs" = 12500,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "alpha_thresh" = 1 / qchisq(1 - (0.05 / 104), df = 1),
  "flist" = flist,
  "lr" = 0.05,
  "err_sig" = 1,
  "convergence_crit" = 1e-7,
  "ttsplit" = 4/5,
  "batch_size" = NULL,
  "stop_k" = 100,
  "stop_streak" = 25,
  "burn_in" = 25e4 # 5E5,
)
set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))



## generate data ----
sim_seed <- sim_params$sim_seeds[1]
set.seed(sim_seed)
torch_manual_seed(sim_seed)

simdat <- sim_func_data(
  n_obs = sim_params$n_obs,
  d_in = sim_params$d_in,
  flist = sim_params$flist,
  err_sigma = sim_params$err_sig
)

simdat_df <- data.frame(
  "y" = as_array(simdat$y),
  "x" = as_array(simdat$x)
)

true_inclusion <- c(
  rep(TRUE, times = length(flist)),
  rep(FALSE, times = sim_params$d_in - length(flist))
)


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


library(spikeslab)
library(spikeslabGAM)




# BART ----




