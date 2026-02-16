##################################################
## Project:   sparseVCBART data
## Date:      Feb 10, 2026
## Author:    Arnie Seong
##################################################

# sparseVCBART paper: https://arxiv.org/pdf/2510.08204
# original VCBART paper: https://arxiv.org/pdf/2003.06416

library(here)
library(latex2exp)
library(tidyverse)
library(MASS)

source(here("Rcode", "analysis_fcns.R"))

# EFFECT MODIFIER FCNS ----
beta_0 <- function(Z){
  z1 <- Z[, 1]
  z2 <- Z[, 2]
  3*z1 + 
    (sin(pi * z1)) * (2 - 5 * (z2 > 0.5)) - 
    2 * (z2 > 0.5)
}

# beta_1 Incorrect in sparseVCBART paper ---_
#   - plots don't match formula
#   - changed 
# beta_1 <- function(Z){
#   # below is the (incorrect) formula from sparseVCBART paper
#   # plotting this function doesn't match their plot at all
#   # https://arxiv.org/pdf/2510.08204
#   # plot clearly has some type of periodic function in it
#   z1 <- Z[, 1]
#   (3 - 3*z1^2)*(z1 > 0.6) - 10*sqrt(z1)*(z1 < 0.25)
# }

beta_1 <- function(Z){
  # this is function beta_2(z) from the OG VCBART paper
  # https://arxiv.org/pdf/2003.06416
  z1 <- Z[, 1]
  (3 - 3*(z1^2)*cos(6*pi*z1))*(z1 > 0.6) - 10*sqrt(z1)*(z1 < 0.25)
}

beta_2 <- function(Z) {1}

beta_3 <- function(Z){
  # matches beta_4(z) from OG paper
  z1 <- Z[, 1]
  z2 <- Z[, 2]
  z3 <- Z[, 3]
  z4 <- Z[, 4]
  z5 <- Z[, 5]
  10*sin(pi*z1*z2) + 20*((z3-0.5)^2) + 10*z4 + 5*z5
}

bfcns_list <- list(
  "beta_0" = beta_0,
  "beta_1" = beta_1,
  "beta_2" = beta_2,
  "beta_3" = beta_3
)




# GENERATE DATA ----
# experiments
#  R = 20 effect modifiers z_i ~ Unif[0,1]
#  x ~ N_p(0, \Sigma), \Sigma_{ij} = 0.5^{|i-j|}
#  \epsilon ~ N(0, 1)
#  1k training obs, 200 test
# experiment 1: p = 3 covariates
# experiment 2: p = 50

library(MASS)

## make X vars ----
corr_fcn <- function(i, j) {0.5^(abs(i-j))}

make_Covmat <- function(p, covar_fcn){
  Sigma <- matrix(NA, nrow = p, ncol = p)
  for (i in 1:p){
    for (j in 1:p){
       Sigma[i, j] <- covar_fcn(i, j)
    }
  }
  return(Sigma)
}

gen_Eydat_sparseVCBART <- function(
  n_obs = 1e3,
  p = 50,
  R = 20,
  covar_fcn = corr_fcn,
  beta_0,
  beta_1,
  beta_2,
  beta_3
){
  # in sparseVCBART paper, Z1:Z5 important, X1:X3 important
  Covmat <- make_Covmat(p, covar_fcn = corr_fcn)
  X <- as.data.frame(
    mvrnorm(n = n_obs, mu = rep(0, p), Sigma = Covmat)
  )
  names(X) <- paste0("x", 1:p)
  
  ## make Z vars (effect modifiers)
  Z <- as.data.frame(
    matrix(
      runif(R*n_obs, 0, 1),
      nrow = n_obs,
      ncol = R
    )
  )
  names(Z) <- paste0("z", 1:R)
  
  ## apply fcns
  b0 <- beta_0(Z)
  b1 <- beta_1(Z)
  b2 <- beta_2(Z)
  b3 <- beta_3(Z)
  
  # generate response vector
  Ey <- b0 + b1 * rowSums(cbind(b1, b2, b3) * X[, 1:3])
  
  true_covs <- c(
    paste0("x", 1:3),
    paste0("z", 1:5)
  )
  
  df <- as.data.frame(
   cbind(
     Ey,
     Z[, 1:5],
     X[, 1:3],
     Z[, 6:R],
     X[, 4:p]
   )
  )
  
  return(df)
}



n_obs <- 1e3
p <- 50
R <- 20
sig_eps <- 1
mu_eps <- 0

Ey_df <- gen_Eydat_sparseVCBART(
  n_obs,
  p,
  R,
  beta_0 = bfcns_list[1],
  beta_1 = bfcns_list[2],
  beta_2 = bfcns_list[3],
  beta_3 = bfcns_list[4]
)

y <- Ey_df[, 1] + rnorm(n_obs)
covars <- Ey_df[, -1]
# check tensor dimensions and feed to NN


# param counts
param_counts_from_dims(dim_vec = c(R + p, 4, 16, 1))







# PLOTTING ----
# make grids for plotting
resol <- 100 # grid resolution
z_gridvec <- 0:resol/resol
x_gridvec <- -(3*resol):(3*resol) / resol

## beta_0 ----
z11_plotmat <- cbind(z_gridvec, 1)
z10_plotmat <- cbind(z_gridvec, 0)
b0z1_1 <- beta_0(z11_plotmat)
b0z1_0 <- beta_0(z10_plotmat)

data.frame(
  "z2_0" = b0z1_0,
  "z2_1" = b0z1_1,
  "z1" = z_gridvec
) %>% 
  pivot_longer(cols = -3, names_to = "z2val") %>% 
  ggplot() + 
  geom_line(
    aes(y = value, x = z1, color = z2val)
  ) +
  labs(
    title = TeX("$\\beta_0(z)$ ~ $z_1$"),
    subtitle = TeX("fcn differs for $z_2 < 0.5$ and $z_2 > 0.5$")
  )



## beta_1 ----
b1 <- beta_1(z11_plotmat)

data.frame(
  "b1" = b1,
  "z1" = z_gridvec
) %>% 
  ggplot() + 
  geom_line(
    aes(y = b1, x = z1)
  ) +
  labs(
    title = TeX("$\\beta_1(z)$ ~ $z_1$")
  )


# FCNS TO PLOT BETAs FROM NN OUTPUT PREDS ----
# to plot output from our neural network requires reparam, since
# we aren't explicitly modeling effect modifiers
# VC model: 
# y = \beta_0(z1, z2) + \beta_1(z1)x1 + \beta_2(.)x2 + \beta_3(z1, z2, z3, z4, z5)x3
beta_0
# for beta_0, plot y against z1 for fixed values of z2, other covs = 0

beta_1
# depends only on z1.  Plot y/x1 against z1 for fixed values of x1

beta_2
# just plot y ~ x2

beta_3
# plot y~z1*z2, y~z3, y~z4, y~z5 for fixed values of x3

make_b0_pred_df <- function(
    resol = 100, 
    p = 50, 
    z2_vals = c(.25, .75),
    froth = FALSE,
    froth_mu = 0.25,
    froth_sig = 0.05
  ){
  # use this to plot Ey ~ z1, for values of z2 below and above 0.5
  # b0(z1, z2) is an intercept term
  # use froth = TRUE to add some noise to the nuisance covars
  b0_z1 <- rep(1:resol/resol, length(z2_vals))
  b0_z2 <- rep(z2_vals, each = resol)
  if (froth){
    zfroth <- matrix(
      rnorm(3*length(b0_z1), froth_mu, froth_sig),
      ncol = 3
    )
    b0_covars <- cbind(b0_z1, b0_z2, zfroth)
  } else {
    b0_covars <- cbind(b0_z1, b0_z2, 0, 0, 0)    
  }

  colnames(b0_covars) <- paste0("z", 1:5)
  
  if (froth){
    zero_mat <- matrix(
      rnorm(p*length(b0_z1), froth_mu, froth_sig),
      ncol = p
    )
  } else {
    zero_mat <- matrix(0, nrow = nrow(b0_covars), ncol = p)    
  }

  colnames(zero_mat) <- paste0("x", 1:p) 
  return(
    as.data.frame(
      cbind(b0_covars, zero_mat)
    )
  )
}

make_b1_pred_df <- function(
  resol = 100,
  x1_vals = c(-2, -1, 0, 1, 2),
  p = 50,
  froth = FALSE,
  froth_mu = 0.25,
  froth_sig = 0.05
){
  # Use this to plot Ey/x1 against z1 for fixed values of x1
  # beta_1 only depends on z1.
  # use froth = TRUE to add some noise to the nuisance covars
  b1_z1 <- rep(1:resol/resol, length(x1_vals))
  b1_x1 <- rep(x1_vals, each = resol)
  b1_covars <- cbind(b1_z1, 0, 0, 0, 0, b1_x1, 0, 0)
  colnames(b1_covars) <- c(paste0("z", 1:5), paste0("x", 1:3))
  if (froth){
    froth_mat <- matrix(
      rnorm(6 * length(b1_z1), froth_mu, froth_sig),
      ncol = 6
    )
    b1_covars[, c(2:5, 7:8)] <- froth_mat
    zero_mat <- matrix(
      rnorm((p-3)*length(b1_z1), froth_mu, froth_sig),
      ncol = p-3
    )
  } else {
    zero_mat <- matrix(0, nrow = nrow(b1_covars), ncol = p-3)
  }
  colnames(zero_mat) <- paste0("x", 4:p)
  
  return(
    as.data.frame(
      cbind(b1_covars, zero_mat)
    )
  )
}



make_b2_pred_df <- function(
    resol = 100,
    p = 50,
    froth = FALSE,
    froth_mu = 0.25,
    froth_sig = 0.05
){
  # Use this to plot Ey/x2 ~ x2
  # beta_2 = 1 (i.e. does not depend on any z)
  b2_x2 <- rep(1:resol/resol)
  b2_covars <- cbind(0, 0, 0, 0, 0, 0, b2_x2, 0)
  colnames(b2_covars) <- c(paste0("z", 1:5), paste0("x", 1:3))
  
  if (froth){
    froth_mat <- matrix(
      rnorm(7 * length(b2_x2), froth_mu, froth_sig),
      ncol = 7
    )
    b2_covars[, c(1:6, 8)] <- froth_mat
    zero_mat <- matrix(
      rnorm((p-3)*length(b2_x2), froth_mu, froth_sig),
      ncol = p-3
    )
  } else {
    zero_mat <- matrix(0, nrow = nrow(b2_covars), ncol = p-3)
  }
  
  colnames(zero_mat) <- paste0("x", 4:p) 
  return(
    as.data.frame(
      cbind(b2_covars, zero_mat)
    )
  )
}


make_b0_pred_df(froth = TRUE)
make_b1_pred_df(froth = TRUE)
make_b2_pred_df(froth = TRUE)
