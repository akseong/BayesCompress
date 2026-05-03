# single sim examination

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


# revert to klcorrected
klc_5x16_orig_p100_mcor.25_1000obs_326454
klc_5x16_orig_p100_mcor.25_ # true sig .88
nfdsmallbias_mutcorr0_5x162000obs_155447 # .5886 1 FP
nfdsmallbias_mutcorr.5_5x162000obs_155447 # .6329 all good
nfdsmallbias_mutcorr.5_5x161000obs_155447 # true sig .6381 all work well at min KL / min test diff 
klc_5x16_orig_p100_mcor0_1000obs_326454  # true sig .832 tau correction OK at 50, 200





# was using torch_horseshoe_smallbias
# but using torch_horseshoe_opus for opus marked sims
# 2fc, 3det
# orig fcns, mut_corr 0.5
# function(x, round_dig = NULL){
#   -cos(pi/1.5*x[,1]) + cos(pi*x[,2]) + sin(pi/1.2*x[,2])
#   + abs(x[,3])^(.75) - x[,4]^2/4
# }
det3_5x16_orig_p50_5000obs_326454 #(8-top) didn't converge (test/train far); not well-calibrated
det3_5x16_orig_p50_2000obs_326454 det3_5x16_origmodfcn_p20_2000obs_144960 # 7 loss looks good, but only tau well-calibrated; again, not really converged?
det3_5x16_orig_p50_1000obs_326454 det3_5x16_origmodfcn_p20_1000obs_144960 # 6 didn't converge
opus_5x16_orig_p20_mcor.25_5000obs_144960 opus_5x16_orig_p20_mcor.25_5000obs_326454# (3rd from top) no convergence
opus_5x16_orig_p20_mcor.25_2000obs_144960 opus_5x16_orig_p20_mcor.25_2000obs_326454# (4th) no convergence
opus_5x16_orig_p20_mcor.25_1000obs_144960 opus_5x16_orig_p20_mcor.25_1000obs_326454# (5th)no convergence.  Actually not bad if take 60k iteration (losses are tightest)
opus_5x16_orig_p20_mcor.5_1000obs_144960 opus_5x16_orig_p20_mcor.5_1000obs_326454 #(2nd from top)  bad also shit
opus_5x16_orig_p20_mcor0_1000obs_144960 opus_5x16_orig_p20_mcor0_1000obs_326454 #(top)  both shit




# function(x, round_dig = NULL){
#   -cos(pi/1.5*x[, 1])*(x[,1]>0) - (x[,1]<0)
#   + cos(pi/2*x[,2])*(x[,2]<0) + sin(pi/1.5*x[,2])*(x[,2]>0)
#   - x[, 3]/(1 + x[,4]^2) + 1 / (1 + 2*x[,5]*(x[,5]>0))
# }
# mut_corr 0.5
# 3fc orig fcns
hshoe_3x32_origmodfcn_p20_2000obs_326454 # (1/bottom sim)  gets x3, x5; only both tau_compsn good
hshoe_3x32_origmodfcn_p20_5000obs_326454 # (2) gets x3, x4, x5;  only both tau_compsn good
# 2fc, 3det
det3_5x16_origmodfcn_p20_1000obs_326454  # (3) gets 3, 4, 5; tau, compsn, both well-calibrated
det3_5x16_origmodfcn_p20_2000obs_326454  # (4) gets x3, x5; tau, compsn well-calibrated
det3_5x16_origmodfcn_p20_5000obs_326454  # (5) gets 3, 4, 5; tau NOT well-calibrated, others good



# 2fc, 3det
# function(x, round_dig = NULL){
#   -cos(pi/1.5*x[, 1]*(x[,1]>0)) 
#   + cos(pi*x[,2]) + sin(pi/1.2*x[,2]*(x[,2]>0)) 
#   - 2*x[, 3]/(1 + x[,4]^2) + 1 / (1 + 2*x[,5]*(x[,5]>0))
# }
# mut_corr 0.5
det1_5x16_origmodfcn_p20_2000obs_144960  # gets 3, 4, 5, but has FPs; tau and tau+sn not well-calibrated
det1_5x16_origmodfcn_p20_5000obs_144960  # gets 3, 4, 5; tau not well-calibrated
det1_5x8_origmodfcn_p20_2000obs_144960   # gets 3, 4, 5; tau not well-calibrated
det1_5x8_origmodfcn_p20_2000obs_326454   # gets 3, 4, 5; tau not well-calibrated; sn also not too good (too large)

# 2fc, 3det
# function(x, round_dig = NULL){
#   (sin(pi*x[, 1]/2))*(x[, 2]>0) - (x[, 2]<0) + x[,1]/2*(x[,1] < 0)
#   + x[,3]/(1 + x[,4] + x[, 5]*(x[, 5]>0))
# }
# mut_corr 0.5
det1_5x8_fcn1p20_500obs_144960 # real bad, didn't converge
det1_5x8_fcn1p20_1000obs_144960 # also real bad, didnt converge?
det1_5x8_fcn1p20_2000obs_144960 # also real bad, didnt converge?
det1_5x8_fcn1p20_500obs_326454 # also real bad, didnt converge?
det1_5x8_fcn1p20_1000obs_326454 # also real bad, didnt converge?
det1_5x8_fcn1p20_2000obs_326454 # also real bad, didnt converge?
# THIS FUNCTION SUCKS


# load
stem <- here::here("sims", "results", "nfdsmallbias_mutcorr.5_5x162000obs_155447")
# extract sim information from first in series
first_sim <- paste0(stem, ".RData")
first_mod <- paste0(stem, ".pt")

load(first_sim)
nn_mod <- torch_load(first_mod)
nn_mod$children
sim_params <- sim_res$sim_params
sim_params$n_obs
sim_params$mut_corr
sim_params$meanfcn
show_vars = paste0("x", 1:4)
true_vec = rep(0, sim_params$d_in)
true_vec[1:4] <- 1 

kmat_global <- kappamat_from_sim_res(sim_res)
kmat_local <- kappamat_from_sim_res(sim_res, type = "local")
kmat_tau <- sim_res$kappa_tc_mat
kmat_sn <- sim_res$kappa_sn_mat
loss_mat <- sim_res$loss_mat

which(loss_mat[, 4] == min(loss_mat[, 4]))
which(abs(loss_mat[, 3] - loss_mat[, 2]) == min(abs(loss_mat[, 3] - loss_mat[, 2])))
order(loss_mat[, 4]) #min KL
order(abs(loss_mat[, 3] - loss_mat[, 2])) 


loss_pltfcn(sim_res, burn = 0)$long_df %>% 
  ggplot(
    aes(
      y = value,
      x = epoch,
      color = loss_type
    )
  ) + 
  geom_line() + 
  labs(
    title = paste0(
      "loss vs epochs"
    )
  )

loss_mat[40:50,]


varmat_pltfcn(
  sim_res$alpha_mat,
  y_name = "alpha",
  burn = 0,
  show_vars = show_vars
)$all_vars_plt

varmat_pltfcn(
  kmat_local,
  y_name = "LOCAL kappa",
  burn = 0,
  show_vars = show_vars
)$all_vars_plt

varmat_pltfcn(
  sim_res$kappa_tc_mat,
  y_name = "tau-corrected kappas",
  burn = 0,
  show_vars = show_vars
)$all_vars_plt

if (!is.null(sim_res$kappa_sn_mat)){
varmat_pltfcn(
  sim_res$kappa_sn_mat,
  y_name = "sncomp-corrected kappas",
  burn = 0,
  show_vars = show_vars
)$all_vars_plt
}


lmrow <- 100
kmat_local[lmrow, ]

err_by_max_bfdr(kmat_local[lmrow, ], true_vec)$plt_fdrs +
  labs(
    subtitle = "using local kappas as dropout probabilities"
  )

err_by_max_bfdr(get_kappas_sntau(nn_mod, print_vals = TRUE), true_vec)$plt_fdrs +
  labs(
    subtitle = "FC2 & composite specnorm-corrected kappas as dropout probabilities"
  )

err_by_max_bfdr(kmat_tau[lmrow, ], true_vec)$plt_fdrs +
  labs(
    subtitle = "tau-corrected kappas as dropout probabilities"
  )

err_by_max_bfdr(kmat_sn[lmrow, ], true_vec)$plt_fdrs +
  labs(
    subtitle = "composite specnorm-corrected kappas as dropout probabilities"
  )

