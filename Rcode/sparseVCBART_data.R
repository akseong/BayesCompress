##################################################
## Project:   sparseVCBART data
## Date:      Feb 10, 2026
## Author:    Arnie Seong
##################################################

# sparseVCBART paper: https://arxiv.org/pdf/2510.08204
# original VCBART paper: https://arxiv.org/pdf/2003.06416

library(latex2exp)
library(ggplot2)
library(MASS)

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
mild_corr_fcn <- function(i, j) {0.5^(abs(i-j))}

make_Covmat <- function(p, covar_fcn){
  Sigma <- matrix(NA, nrow = p, ncol = p)
  for (i in 1:p){
    for (j in 1:p){
       Sigma[i, j] <- covar_fcn(i, j)
    }
  }
  return(Sigma)
}

n_obs <- 1e3
p <- 50
R <- 20
sig_eps <- 1
mu_eps <- 0
Covmat <- make_Covmat(p, covar_fcn = mild_corr_fcn)
X <- mvrnorm(n = n_obs, mu = rep(0, p), Sigma = Covmat)


## make Z vars (effect modifiers)

Z <- matrix(
  runif(R*n_obs, 0, 1),
  nrow = n_obs,
  ncol = R
)


## apply fcns
b0 <- beta_0(Z)
b1 <- beta_1(Z)
b2 <- beta_2(Z)
b3 <- beta_3(Z)

Ey <- b0 + 
  b1 * rowSums(cbind(b1, b2, b3) * X[, 1:3])

y <- Ey + rnorm(n = length(Ey), mu_eps, sig_eps)


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



