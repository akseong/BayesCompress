##################################################
## Project:   sparseVCBART data generation & plotting fcns
## Date:      Feb 17, 2026
## Author:    Arnie Seong
##################################################

library(MASS)

# EFFECT MODIFIER FCNS----
beta_0 <- function(Z){
  z1 <- Z[, 1]
  z2 <- Z[, 2]
  3*z1 + 
    (sin(pi * z1)) * (2 - 5 * (z2 > 0.5)) - 
    2 * (z2 > 0.5)
}

# beta_1 Incorrect in sparseVCBART paper ---
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



## MAKE X VARS FCNS ----
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


# GENERATE MEAN FUNCTION DATA----
gen_Eydat_sparseVCBART <- function(
    n_obs = 1e3,
    p = 50,
    R = 20,
    covar_fcn = corr_fcn,
    beta_0 = beta_0,
    beta_1 = beta_1,
    beta_2 = beta_2,
    beta_3 = beta_3
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
  Ey <- b0 + rowSums(cbind(b1, b2, b3) * X[, 1:3])
  
  true_covs <- c(
    paste0("x", 1:3),
    paste0("z", 1:5)
  )
  
  df <- as.data.frame(cbind(Ey, Z, X))
  return(df)
}


# PLOTTING TRUE BETA FCNS----
# make grids for plotting
plot_b0_true <- function(resol = 100, b0 = beta_0){
  require(tidyverse)
  require(latex2exp)
  z_gridvec <- 0:resol/resol
  z11_plotmat <- cbind(z_gridvec, 1)
  z10_plotmat <- cbind(z_gridvec, 0)
  b0z1_1 <- b0(z11_plotmat)
  b0z1_0 <- b0(z10_plotmat)
  
  plt <- data.frame(
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
  
  return(plt)
}


plot_b1_true <- function(resol = 100, b1 = beta_1){
  require(tidyverse)
  require(latex2exp)
  z_gridvec <- 0:resol/resol
  z11_plotmat <- cbind(z_gridvec, 1)
  b1 <- beta_1(z11_plotmat)
  
  plt <- data.frame(
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
  
  return(plt)
}



# MAKE PRED PLOT DFs----
# fcns make dataframes used to plot predictions
make_b0_pred_df <- function(
    resol = 100, 
    p = 50,
    R = 20,
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
    zfill <- matrix(
      rnorm((R-2)*length(b0_z1), froth_mu, froth_sig),
      ncol = R-2
    )
  } else {
    z_fill <- matrix(0, nrow = length(b0_z1), ncol = R-2)
  }
  b0_Z <- cbind(b0_z1, b0_z2, z_fill0)    
  colnames(b0_Z) <- paste0("z", 1:R)
  
  if (froth){
    zero_mat <- matrix(
      rnorm(p*length(b0_z1), froth_mu, froth_sig),
      ncol = p
    )
  } else {
    zero_mat <- matrix(0, nrow = length(b0_z1), ncol = p)    
  }
  
  colnames(zero_mat) <- paste0("x", 1:p)
  df <- as.data.frame(cbind(b0_Z, zero_mat))
  return(df)
}

make_b1_pred_df <- function(
    resol = 100,
    x1_vals = c(-2, -1, 0, 1, 2),
    p = 50,
    R = 20,
    froth = FALSE,
    froth_mu = 0.25,
    froth_sig = 0.05
){
  # Use this to plot Ey/x1 against z1 for fixed values of x1
  # beta_1 only depends on z1.
  # use froth = TRUE to add some noise to the nuisance covars
  b1_z1 <- rep(1:resol/resol, length(x1_vals))
  b1_x1 <- rep(x1_vals, each = resol)
  
  if (froth){
    z_fill <- matrix(
      rnorm((R-1) * length(b1_z1), froth_mu, froth_sig),
      ncol = R-1
    )
    x_fill <- matrix(
      rnorm((p-1) * length(b1_z1), froth_mu, froth_sig),
      ncol = p-1
    )
  } else {
    z_fill <- matrix(0, nrow = length(b1_z1), ncol = R-1)
    x_fill <- matrix(0, nrow = length(b1_z1), ncol = p-1)
  }
  b1_dat <- cbind(b1_z1, z_fill, b1_x1, x_fill)
  colnames(b1_dat) <- c(paste0("z", 1:R), paste0("x", 1:p))
  df <- as.data.frame(b1_dat)
  return(df)
}



make_b2_pred_df <- function(
    resol = 100,
    p = 50,
    R = 20,
    froth = FALSE,
    froth_mu = 0.25,
    froth_sig = 0.05
){
  # Use this to plot Ey/x2 ~ x2
  # beta_2 = 1 (i.e. does not depend on any z)
  b2_x2 <- rep(1:resol/resol)
  
  if (froth){
    z_fill <- matrix(
      rnorm(R * length(b2_x2), froth_mu, froth_sig),
      ncol = R
    )
    x_fill <- matrix(
      rnorm((p-1) * length(b2_x2), froth_mu, froth_sig),
      ncol = p-1
    )
  } else {
    z_fill <- matrix(0, nrow = length(b2_x2), ncol = R)
    x_fill <- matrix(0, nrow = length(b2_x2), ncol = p-1)
  }

  b2_dat <- cbind(
    z_fill, 
    x_fill[, 1], 
    b2_x2, 
    x_fill[, 2:(p-1)]
    )
  colnames(b2_dat) <- c(paste0("z", 1:R), paste0("x", 1:p))
  df <- as.data.frame(b2_dat)
  return(df)
}


# TESTING----
n_obs <- 1e3
p <- 50
R <- 20
sig_eps <- 1
mu_eps <- 0

bfcns_list <- list(
  "beta_0" = beta_0,
  "beta_1" = beta_1,
  "beta_2" = beta_2,
  "beta_3" = beta_3
)

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


plot_b0_true()
plot_b1_true()






## check plotting strat for b0 ----
b0_pred_df <- make_b0_pred_df()
head(b0_pred_df)

b0 <- beta_0(b0_pred_df[, 1:R])
plot(b0 ~ b0_pred_df[,1])

b1 <- beta_1(b0_pred_df[, 1:R])
b2 <- beta_2(b0_pred_df[, 1:R])
b3 <- beta_3(b0_pred_df[, 1:R])

Ey <- b0 + rowSums(cbind(b1, b2, b3) * b0_pred_df[, 1:p + R])
plot(Ey ~ b0_pred_df[, 1])


## check plotting strat for b1 ----
# Ey/x1 ~ z1
# beta_3 = 10*sin(pi*z1*z2) + 20*((z3-0.5)^2) + 10*z4 + 5*z5
#   so to recover b1, set z2 = 0, z3 = .5, z4=0, z5=0,
#   or just x3 = 0
# beta_2 = 1, so just set x2 = 0
# beta_0 =   3*z1   +   (sin(pi * z1)) * (2 - 5 * (z2 > 0.5))   -   2 * (z2 > 0.5)
#   also depends on z1; does not depend on any x value, 
#   i.e. cannot separate beta_1 from beta_0
#   - guess just plot against the truth here
# Maybe can alleviate this by placing an intercept term in design matrix?
#    - if doing this, need to omit bias term in first layer.
b1_pred_df <- make_b1_pred_df()
head(b1_pred_df)

b1 <- beta_1(b1_pred_df[, 1:R])
plot(b1 ~ b1_pred_df[,1])

b0 <- beta_0(b1_pred_df[, 1:R])
b2 <- beta_2(b1_pred_df[, 1:R])
b3 <- beta_3(b1_pred_df[, 1:R])

cbind(b1, b2, b3) * b1_pred_df[, 1:3 + R]
b0

Ey <- b0 + rowSums(cbind(b1, b2, b3) * b1_pred_df[, 1:3 + R])

plot(Ey/b1_pred_df[,(R+1)] ~ b1_pred_df[, 1])





## check plotting strat for b2 ----
# Ey/x2 ~ x2 seems fines
b2_pred_df <- make_b2_pred_df()
head(b2_pred_df)

b2 <- beta_2(b2_pred_df[, 1:R])
plot(b2 ~ b2_pred_df[,1])

b0 <- beta_0(b2_pred_df[, 1:R])
b1 <- beta_1(b2_pred_df[, 1:R])
b3 <- beta_3(b2_pred_df[, 1:R])

Ey <- b0 + rowSums(cbind(b1, b2, b3) * b2_pred_df[, 1:p + R])
plot(Ey/b2_pred_df[, R+2] ~ b2_pred_df[, R+2])






