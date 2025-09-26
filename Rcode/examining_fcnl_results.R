
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
thresh_calc <- function(t1_rate = 0.05, bonf = 104, df = 1){
    1 / qchisq(1 - (t1_rate / bonf), df = df)
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
w1 <- 1/alphas_1

sum(alphas_1)
rel_w <- w1 / mean(w1)
round(rel_w, 3)

w1 > 1 / thresh_calc(df = 1)


## BFDR ----
# BFDR chooses well when set very low (e.g. 0.001), poorly when set any higher than 0.003,
# very poorly when = 0.05 (chooses 27 spurious)
BFDR <- function(dropout_probs, eta, show_decisions = TRUE){
  delta_vec <- 1 * (dropout_probs <= eta)
  bfdr <- sum((dropout_probs) * delta_vec) / sum(delta_vec)
  if(show_decisions) cat("delta vector: ", delta_vec, "\n ")
  return(bfdr)
}

BFDR_eta_search <- function(dropout_probs, max_rate = 0.05){
  a_sort <- sort(dropout_probs)
  bfdrs <- sapply(a_sort, function(X) BFDR(dropout_probs, eta = X, show_decisions = FALSE))
  eta = a_sort[max(which(bfdrs <= max_rate))]
  eta
}

eta <- BFDR_eta_search(alphas_1, max_rate = 0.005)
BFDR(alphas_1, eta)
sum(alphas_1 <= eta)

# but alphas are not dropout probs.  Instead, more like quantiles of Chi-sq(1) (posterior Wald)
dropout_probs <- pchisq(alphas_1, df = 1)
eta <- BFDR_eta_search(dropout_probs, max_rate = 0.01)
BFDR(dropout_probs, eta)
d_i <- dropout_probs <= eta
sum(d_i)
# nice!!!

# oh damn.  actually, more like quantiles of INVERSE chi-sq(1)
dropout_probs <- 1 - pchisq(1/alphas_1, df = 1)
eta <- BFDR_eta_search(dropout_probs, max_rate = 0.01)
BFDR(dropout_probs, eta)
d_i <- dropout_probs <= eta
sum(d_i)
# booo.  47

eta <- BFDR_eta_search(dropout_probs, max_rate = 0.00001)
BFDR(dropout_probs, eta)
d_i <- dropout_probs <= eta
sum(d_i)
# still no good, even with v. small bfdr rate


#### using shrinkage factor Kappa ----
at <- as_array(nn_model$fc1$atilde_mu)
bt <- as_array(nn_model$fc1$btilde_mu)
sa <- as_array(nn_model$fc1$sa_mu)
sb <- as_array(nn_model$fc1$sb_mu)

at_var <- exp(as_array(nn_model$fc1$atilde_logvar))
bt_var <- exp(as_array(nn_model$fc1$btilde_logvar))
sa_var <- exp(as_array(nn_model$fc1$sa_logvar))
sb_var <- exp(as_array(nn_model$fc1$sb_logvar))

ln_mode <- function(mu, var){
  exp(mu-var)
}

ln_mean <- function(mu, var){
  exp(mu + var/2)
}


m_at <- ln_mode(at, at_var)
m_bt <- ln_mode(bt, bt_var)
m_sa <- ln_mode(sa, sa_var)
m_sb <- ln_mode(sb, sb_var)

e_at <- ln_mean(at, at_var)
e_bt <- ln_mean(bt, bt_var)
e_sa <- ln_mean(sa, sa_var)
e_sb <- ln_mean(sb, sb_var)

zsq <- abs(at*bt*sa*sb)
zsq_lower <- abs((abs(at)-2*at_var)*(abs(bt)-2*bt_var)*(abs(sa)-2*sa_var)*(abs(sb)-2*sb_var))
zsq_mode <- m_at * m_bt * m_sa * m_sb
zsq_mean <- e_at * e_bt * e_sa * e_sb
round(zsq_mode, 3)
round(zsq_mean, 3)

kappa <- 1 / (1 + zsq_mean)

eta <- BFDR_eta_search(kappa, max_rate = 0.01)
BFDR(kappa, eta)
sum(kappa <= eta)

eta <- BFDR_eta_search(kappa, max_rate = 0.05)
BFDR(kappa, eta)
sum(kappa <= eta)

eta <- BFDR_eta_search(kappa, max_rate = 0.1)
BFDR(kappa, eta)
sum(kappa <= eta)


#### using shrinkage factor Kappa based only on local shrinkage (i.e. tau = 1) ----
#### WORKS PRETTY DAMN WELL.  
# ztil = (tilde{z})^2
ztil_mode <- m_at * m_bt
ztil_mean <- e_at * e_bt
ztil_mode - ztil_mean
#### works better with mode, which makes sense (VI is mode-seeking).


kappa_til <- 1/(1 + ztil_mode)
round(kappa_til, 3)
round(kappa, 3)

eta <- BFDR_eta_search(kappa_til, max_rate = 0.01)
BFDR(kappa_til, eta)
sum(kappa_til <= eta)

eta <- BFDR_eta_search(kappa_til, max_rate = 0.05)
BFDR(kappa_til, eta)
sum(kappa_til <= eta)

eta <- BFDR_eta_search(kappa_til, max_rate = 0.1)
BFDR(kappa_til, eta)
sum(kappa_til <= eta)




## Decision Theory ----

# Loss fcn experiments:
dmat <- rbind(
  "none" = 0,
  "TPonly" = alphas < 0.001,
  "FP1" = sum(alphas < 0.015),
  "FP2" = alphas < 0.02,
  "FP5" = alphas < 0.03,
  "FP23" = alphas < 0.1,
  "FP52" = alphas < 0.5,
  "FPonly" = c(0,0,0,0, rep(1, 100))
)
none <- dmat[1, ]
tp <- dmat[2, ]
fp23 <- dmat[3, ]
fp52 <- dmat[4, ]
fp <- dmat[5,]

# 1/alphas --- large for TPs
x <- sort(alphas)
y <- 1/(x)
plot(y~x)

# 1/(1 - alphas) --- large for FPs
y <- 1/(1-x)
plot(y~x)

y <- -1/x + 1/(1-x)
plot(y~x)

y <- log(x)

# positives: 1 - alpha
# negatives: alpha
# penalize number chosen
lf <- function(a, d_i, c1=20, c2=100){
  # uses 1/alpha (v. high TP, moderately high FPs) and 1/(1-alpha) (high FP)
  # penalize # positives chosen
  - sum(d_i * (1/a)) + c1*sum(!d_i*(1/(1-a))) + c2*sum(d_i)
}

lf2 <- function(a, d_i, c1=20, c2=100){
  - sum(d_i * (1-a)) + c1*sum((1-d_i)*(a)) + c2*sum(d_i)
}


c1=1
c2=100
d_i <- alphas_1 < .03
sort(alphas_1)
lf(alphas_1, d_i, c1, c2)
lf2(alphas_1, d_i, 1, 1)
order(alphas_1)
c2 / (1 + c1)
apply(dmat, 1, function(X) lf(alphas_1, d_i = X, c1=1, c2=100))
apply(dmat, 1, function(X) lf2(alphas_1, d_i = X, c1=5, c2=1))


# posterior: ----
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

