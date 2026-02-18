##################################################
## Project:   sparseVCBART sim
## Date:      Feb 18, 2026
## Author:    Arnie Seong
##################################################


#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe_klcorrected.R"))
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))
source(here("Rcode", "sparseVCBART_fcns.R"))

## data generation----
n_obs <- 1e3
p <- 50
R <- 20
sig_eps <- 1
mu_eps <- 0

# beta_j(z) functions in sparseVCBART_fcns.R
bfcns_list <- list(
  "beta_0" = beta_0,
  "beta_1" = beta_1,
  "beta_2" = beta_2,
  "beta_3" = beta_3
)

# note that b1 and b0 are not really separable when looking at y
# without modeling the effect modifiers directly as VCBART does
plot_b0_true(resol = 100, b0 = bfcns_list$beta_0)
plot_b1_true(resol = 100, b1 = bfcns_list$beta_1)


corr_fcn <- function(i, j) {0.5^(abs(i-j))}

Ey_df <- gen_Eydat_sparseVCBART(
  n_obs,
  p,
  R,
  covar_fcn = corr_fcn,
  beta_0 = bfcns_list$beta_0,
  beta_1 = bfcns_list$beta_1,
  beta_2 = bfcns_list$beta_2,
  beta_3 = bfcns_list$beta_3
)

range(Ey_df[, 1])
head(Ey_df)
true_covs <- c(
  paste0("x", 1:3),
  paste0("z", 1:5)
)

# param counts
source(here::here("Rcode", "analysis_fcns.R"))
param_counts_from_dims(dim_vec = c(R + p, 4, 16, 1))







