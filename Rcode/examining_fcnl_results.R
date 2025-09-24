
#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))


if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}

# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) - (abs(x))
# flist = list(fcn1, fcn2, fcn3, fcn4)
fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x
fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) -x^2 / 2 -3
fcn4 <- function(x) - log(abs(x) + 1e-3)
flist = list(fcn1, fcn2, fcn3, fcn4)
plot_datagen_fcns(flist)
# 
# xshow <- seq(-3, 3, length.out = 100)
# yshow <- sapply(flist, function(fcn) fcn(xshow))
# df <- data.frame(
#   "f1" = yshow[, 1],
#   "f2" = yshow[, 2],
#   "f3" = yshow[, 3],
#   "f4" = yshow[, 4],
#   "x"  = xshow
# )
# df %>% 
#   pivot_longer(cols = -x, names_to = "fcn") %>% 
#   ggplot(aes(y = value, x = x, color = fcn)) +
#   geom_line() + 
#   labs(title = "functions used to create data")




## sim_params
#    check whenever changing setting (testing / single vs parallel, etc) ##
#           n_sims, verbose, want_plots, train_epochs

sim_params <- list(
  "sim_name" = "hshoe, tau fixed, 2 layers 16 8 (worked before) nobatching, fcnal data.  ",
  "seed" = 2168,
  "n_sims" = 1, 
  "train_epochs" = 5e5, # 15E5,
  "report_every" = 1e4, # 1E4,
  "use_cuda" = use_cuda,
  "d_in" = 104,
  "d_hidden1" = 16,
  "d_hidden2" = 8,
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

nn_model <- torch::torch_load(here::here("sims", "results", "fcnl_hshoe_mod_12500obs_398060.pt"))

round(exp(as_array(nn_model$fc1$weight_logvar)), 3)

post_weight_mu1 <- as_array(nn_model$fc1$compute_posterior_param()$post_weight_mu)
post_weight_var1 <- as_array(nn_model$fc1$compute_posterior_param()$post_weight_var)
round(post_weight_mu1, 3)
round(post_weight_var1, 3)

alphas_1 <- as_array(nn_model$fc1$get_dropout_rates())
round(alphas_1, 3)
sum(alphas_1 < sim_params$alpha_thresh)

global_alphas_1 <- as_array(nn_model$fc1$get_dropout_rates("global"))
marg_alphas_1 <- as_array(nn_model$fc1$get_dropout_rates("marginal"))
round(marg_alphas_1, 3)
sum(marg_alphas_1 < sim_params$alpha_thresh)



# # # # # # # # # # # # # # # # # # # # # # # # #
## Degrees of Freedom ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# many, many more observations here, so should change the chisq dist degrees of freedom
# was using df = 1 earlier because only had 100 obs, and many more parameters.
# now, had 10k obs, which is more than the # parameters

# how to determine degrees of freedom?  

## df = number of trainable params used to calculate alpha (local dropout rate)----
# local alpha = exp{  1/4 * [exp(atilde_logvar) + exp(btilde_logvar)]  - 1}
# so just atilde_logvar and btilde_logvar (atilde * btilde = ztilde^2)
# both are atilde_logvar and btilde_logvar have length d_in (here, 104)

n_obs <- 1e4
n_alpha_params <- 104*2

# exceedingly small
1 / qchisq(1 - (0.05 / 104), df = (n_obs - n_alpha_params))

alphas_1 < 1 / qchisq(1 - (0.05 / 104), df = (n_obs - n_alpha_params))



# or total number of varying params? ----
#   ####  this would mean a network with higher capacity (deeper/wider) would
#   ####  have lower df, which seems WEIRD.
# at higher df (e.g. df > 20), chi square is for all practical purposes flat 
# (median ~ mean = df)
# i.e. higher df --> lower alpha threshold

x <- 1:500/100
chisq_pltdf <- data.frame(
  x,
  df001 = dchisq(x, df = 1),
  df003 = dchisq(x, df = 3),
  df005 = dchisq(x, df = 5),
  df010 = dchisq(x, df = 10),
  df020 = dchisq(x, df = 20)
)


chisq_pltdf %>% 
  pivot_longer(cols = -x, names_to = "df", values_to = "dens") %>% 
  filter(dens < 0.25) %>% 
  ggplot() + 
  geom_line(aes(y = dens, x = x, colour = df)) + 
  ylim(0, 0.25) + 
  labs(title = "Chi-sq densities by df")


bonf <- 104
df <- 1:1000
thresh_calc <- function(t1rate = 0.05, bonf = 104, df = 1){
    1 / qchisq(1 - (t1rate / bonf), df = df)
}

df_vec <- 1:1000
thresh_vec <- thresh_calc(df = df_vec)
plot(thresh_vec ~ df_vec, 
     main = "alpha threshold ~ df",
     ylab = "thresh (alpha must be under) for inclusion", xlab = "df",
     type = "l")

thresh_calc(df = 1)
thresh_calc(df = 100)
thresh_calc(df = 1000)

# with many obs, 100 more parameters 
# in network doesn't affect threshold much
thresh_calc(df = 10000)
thresh_calc(df = 9900)


# total number of trainable_params 
n_trainable <- 0
for (param in nn_model$parameters) {
  if (param$requires_grad) {
    n_trainable <- n_trainable + param$numel()
  }
}

thresh_calc(df = n_obs - n_trainable)
alphas_1 < thresh_calc(df = n_obs - n_trainable)



## effective degrees of freedom? ----
# Gao, Jojic 2016: Degrees of freedom in deep neural networks
# https://arxiv.org/abs/1603.09260






























# posterior:
nsamps <- 10000
x <- rlnorm(nsamps, 1, 1)
plot(density(log(x)))

rlnorm(nsamps, -1, 1)


E_s <- E_lognorm(
  mu = (self$sa_mu + self$sb_mu) / 2, 
  logvar = (self$sa_logvar + self$sb_logvar) - log(4)
)

E_ztil <- E_lognorm(
  mu = (self$atilde_mu + self$btilde_mu) / 2, 
  logvar = (self$atilde_logvar + self$btilde_logvar) - log(4)
)

Ez <- E_lognorm(
  mu = (self$atilde_mu + self$btilde_mu + self$sa_mu + self$sb_mu) / 2, 
  logvar = (self$atilde_logvar + self$btilde_logvar + self$sa_logvar + self$sb_logvar) - log(4)
)

z_m = as_array((self$atilde_mu + self$btilde_mu + self$sa_mu + self$sb_mu) / 2)
z_lv = as_array((self$atilde_logvar + self$btilde_logvar + self$sa_logvar + self$sb_logvar) - log(4))
z_sig = exp(z_lv/2)
z_m  # in model, large z_m indicates inclusion (W ~ N(0, z_m^2))

round(as_array(nn_model$fc1$post_weight_mu), 3)[, 1:4]
round(as_array(nn_model$fc1$post_weight_mu), 3)[, 5:10]

