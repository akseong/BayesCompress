##################################################
## Project:   Figs
## Date:      May 11, 2026
## Author:    Arnie Seong
##################################################


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


reconstruct_meanfcndat <- function(
    sim_seed,
    sim_params
){
  set.seed(sim_seed)
  torch_manual_seed(sim_seed)
  
  simdat <- sim_meanfcn_data(
    n_obs = sim_params$n_obs, 
    d_in = sim_params$d_in, 
    mut_corr = sim_params$mut_corr,
    genXfcn = sim_params$genXfcn,
    meanfcn = sim_params$meanfcn,
    err_sigma = sim_params$err_sig, 
    round_dig = sim_params$round_dig,
    standardize = sim_params$standardize
  )
  
  return(simdat)
}


reconstruct_flistdat <- function(
    sim_seed,
    sim_params
){
  set.seed(sim_seed)
  torch_manual_seed(sim_seed)
  
  simdat <- sim_func_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    xlist = NULL,
    err_sigma = sim_params$err_sig,
    use_cuda = sim_params$use_cuda,
    xdist = sim_params$xdist,
    xcov = NULL,
    mut_corr = sim_params$mut_corr,
    standardize = sim_params$standardize
  )
  
  return(simdat)
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


stem <- here::here("final_sims", "results", "hshoesmallbias_mutcorr0.5_5x161000obs_")
modfcns_TF <- grepl("meanfs", stem)
n_sims = 50 

if (modfcns_TF){
  reconstruct_fcn <- reconstruct_meanfcndat
  true_vec <- rep(0, 108)
  true_vec[1:8] <- 1
  fname_suffix <- "modfcns_fig"   
} else {
  reconstruct_fcn <- reconstruct_flistdat
  true_vec <- rep(0, 104)
  true_vec[1:4] <- 1
  fname_suffix <- "origfcns_fig"
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
poss_modnames <- paste0(stem, possible_sim_seeds, ".pt")
exists_TF <- rep(F, length(poss_fnames))
modexists_TF <- rep(F, length(poss_modnames))

for (i in 1:length(poss_fnames)){
  exists_TF[i] <- file.exists(poss_fnames[i])
  modexists_TF[i] <- file.exists(poss_modnames[i])
}

all.equal(exists_TF, modexists_TF)
sum(exists_TF)
if (length(unique(poss_fnames[exists_TF])) < n_sims) {warning("fewer than ", n_sims, " exist")}

sim_fnames <- paste0(stem, possible_sim_seeds[exists_TF], ".RData")[1:n_sims]
mod_fnames <- paste0(stem, possible_sim_seeds[exists_TF], ".pt")[1:n_sims]
sim_seeds <- possible_sim_seeds[exists_TF][1:n_sims]


mse_mat <- matrix(NA, nrow = n_sims, ncol = 3)
colnames(mse_mat) <- c("test_mse", "fcn_mse", "train_sig")

# construct filename ----
load(sim_fnames[1])
nn_mod <- torch_load(paste0(stem, possible_sim_seeds[exists_TF][1], ".pt"))
hshoe_layers <- grepl("fc", names(nn_mod$children))
det_layers <- grepl("det", names(nn_mod$children))
architecture_str <- paste0("hshoe", sum(hshoe_layers), "det", sum(det_layers))

fig_stem <- paste0(
  architecture_str, "_", 
  sim_res$sim_params$n_obs/1000, "k_",
  n_sims, "sims_", fname_suffix
  )

fig_fname <- here::here("final_sims", "compiled", paste0(fig_stem, ".Rdata"))


# load sim ----
s_i = 1
load(sim_fnames[s_i])
nn_mod <- torch_load(mod_fnames[s_i])
# reconstruct data ----
use_cuda <- nn_mod$fc1$atilde_logvar$is_cuda
sim_res$sim_params$use_cuda <- use_cuda
sim_res$sim_params$standardize <- TRUE

simdat <- reconstruct_fcn(sim_seed = sim_seeds[s_i], sim_res$sim_params)

# scale train/test split, remove Ey from simdat ----
n_ttsplit <- sim_res$sim_params$ttsplit * sim_res$sim_params$n_obs
x_train <- simdat$x[1:n_ttsplit, ]
x_test <- simdat$x[(1+n_ttsplit):sim_res$sim_params$n_obs, ]
y_train <- simdat$y[1:n_ttsplit, ]
y_test <- simdat$y[(1+n_ttsplit):sim_res$sim_params$n_obs, ]


# Ey was never scaled.... fuck.  just scale it by the y scales?
# actually, no, this is good. keep Ey and ytest unscaled
if(use_cuda){
  Ey <- simdat$Ey$unsqueeze(2)$cuda()  
} else {
  Ey <- simdat$Ey$unsqueeze(2)
}
Ey_test <- Ey[(1+n_ttsplit):sim_res$sim_params$n_obs, ]

# test_mse
nn_mod$eval()
yhat_test <- nn_mod(x_test)

# unscaled
yhat_test_unsc <- (yhat_test + simdat$y_mean)*simdat$y_sd
mse_test <-  mean((yhat_test_unsc - y_test)^2)
fmse_test <- mean((yhat_test_unsc - Ey_test)^2)

# FUNCTION RECOVERY ----
# run regular forward pass 100 times using data that includes x1:x4 one at a time, mean center each of the generated functions
# plot with data (after mean-centering based on true Ey)
# for one dataset and one trained model for each of 1k, 2k, 5k


# construct x_mat




































# for (s_i in 1:n_sims){
#   load(sim_fnames[s_i])
#   nn_mod <- torch_load(mod_fnames[s_i])
#   # reconstruct data ----
#   use_cuda <- nn_mod$fc1$atilde_logvar$is_cuda
#   sim_res$sim_params$use_cuda <- use_cuda
#   sim_res$sim_params$standardize <- TRUE
#   
#   simdat <- reconstruct_fcn(sim_seed = sim_seeds[s_i], sim_res$sim_params)
#   
#   # scale train/test split, remove Ey from simdat
#   n_ttsplit <- sim_res$sim_params$ttsplit * sim_res$sim_params$n_obs
#   x_train <- simdat$x[1:n_ttsplit, ]
#   x_test <- simdat$x[(1+n_ttsplit):sim_res$sim_params$n_obs, ]
#   y_train <- simdat$y[1:n_ttsplit, ]
#   y_test <- simdat$y[(1+n_ttsplit):sim_res$sim_params$n_obs, ]
#   
#   # Ey was never scaled.... fuck.  just scale it by the y scales?
#   if(use_cuda){
#     Ey_raw <- simdat$Ey$unsqueeze(2)$cuda()  
#   } else {
#     Ey_raw <- simdat$Ey$unsqueeze(2)
#   }
#   Ey <- (Ey_raw - simdat$y_mean)/simdat$y_sd
#   # Ey <- (Ey_raw - mean(Ey_raw))/sd(Ey_raw)
#   Ey_train <- Ey[1:n_ttsplit, ]
#   Ey_test <- Ey[(1+n_ttsplit):sim_res$sim_params$n_obs, ]
#   
#   # test_mse
#   nn_mod$eval()
#   yhat_test <- nn_mod(x_test)
#   
#   test_mse <- mean((yhat_test - y_test)^2)
#   
#   # function mse
#   test_fmse <- mean((yhat_test - Ey_test)^2)
#   
#   mse_mat[s_i, ] <- c(test_mse$item(), test_fmse$item(), sim_res$sim_params$train_sig)
#   # function recovery
# }
# 
# save(mse_mat, file = mse_fname)
# cat_color(paste0("mse_mat file saved to: ", mse_fname))




