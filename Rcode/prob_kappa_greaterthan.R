

a_marg <- as_array(nn_mod$fc1$get_dropout_rates(type = "marginal"))
zscore <- -1/sqrt(a_marg)

ztil_params <- get_ztil_params(nn_mod$fc1)
s_params <- get_s_params(nn_mod$fc1)

a_mu <- ztil_params$at
a_lvar <- ztil_params$at_lvar
b_mu <- ztil_params$bt
b_lvar <- ztil_params$bt_lvar

sa_mu <- s_params$sa
sb_mu <- s_params$sb
sa_lvar <- s_params$sa_lvar
sb_lvar <- s_params$sb_lvar

s2_params <- get_s_params(nn_mod$fc2)
sa2_mu <- s2_params$sa
sb2_mu <- s2_params$sb
sa2_lvar <- s2_params$sa_lvar
sb2_lvar <- s2_params$sb_lvar

ssq <- get_s_sq(nn_mod$fc1)
sqrt(ssq)

zsq <- get_zsq(nn_mod$fc1)
z <- sqrt(zsq)
ztil_sq <- get_ztil_sq(nn_mod$fc1)
ztil <- sqrt(ztil_sq)

sncorr <- get_composite_specnorm(nn_mod)



z_mu <- (a_mu + b_mu + sa_mu + sb_mu)/2
z_var <- (exp(a_lvar) + exp(b_lvar) + exp(sa_lvar) + exp(sb_lvar))/4
z_sd <- sqrt(z_var)

round(pnorm((-log(sncorr)-z_mu) / z_var), 3)   # prob that kappa > 0.5

p_kthresh <- function(mu_z, sig_z, tau_corr = 1, thresh = 0.25){
  z_thresh <- log((1 - thresh)/thresh) / 2 - log(tau_corr)
  pnorm((z_thresh - mu_z) / sig_z)
}



get_p_kgtc <- function(nn_mod, tau_corr = 1, thresh = 0.5){
  a_mu <- as_array(nn_mod$fc1$atilde_mu)
  b_mu <- as_array(nn_mod$fc1$btilde_mu)
  a_lvar <- as_array(nn_mod$fc1$atilde_logvar)
  b_lvar <- as_array(nn_mod$fc1$btilde_logvar)
  
  sa_mu <- as_array(nn_mod$fc1$sa_mu)
  sb_mu <- as_array(nn_mod$fc1$sb_mu)
  sa_lvar <- as_array(nn_mod$fc1$sa_logvar)
  sb_lvar <- as_array(nn_mod$fc1$sa_logvar)
  
  z_mu <- (a_mu + b_mu + sa_mu + sb_mu)/2
  z_var <- (exp(a_lvar) + exp(b_lvar) + exp(sa_lvar) + exp(sb_lvar))/4
  z_sd <- sqrt(z_var)
  p_kthresh(mu_z = z_mu, sig_z = z_sd, tau_corr = tau_corr, thresh = thresh)
}


round(
  p_kthresh(mu_z = (a_mu + b_mu)/2,
          sig_z = (exp(a_lvar) + exp(b_lvar))/4),
  2)



tcorr <- get_composite_specnorm(nn_mod)
tcorr <- tau_correction(nn_mod)
get_kappas(nn_mod$fc1)
cbind(
  1 - round(get_p_kgtc(nn_mod, tau_corr = tcorr, thresh = 0.05), 2),
  1 - round(get_p_kgtc(nn_mod, tau_corr = tcorr, thresh = 0.10), 2),
  1 - round(get_p_kgtc(nn_mod, tau_corr = tcorr, thresh = 0.25), 2),
  1 - round(get_p_kgtc(nn_mod, tau_corr = tcorr, thresh = 0.50), 2),
  1 - round(get_p_kgtc(nn_mod, tau_corr = tcorr, thresh = 0.75), 2),
  1 - round(get_p_kgtc(nn_mod, tau_corr = tcorr, thresh = 0.90), 2)
)





#### based on mode of z: if z ~ LN(mu, sig^2), mode(z) = exp(mu - sig^2)
ztil_params <- get_ztil_params(nn_mod$fc1)
s_params <- get_s_params(nn_mod$fc1)

a_mu <- ztil_params$at
a_lvar <- ztil_params$at_lvar
b_mu <- ztil_params$bt
b_lvar <- ztil_params$bt_lvar

sa_mu <- s_params$sa
sb_mu <- s_params$sb
sa_lvar <- s_params$sa_lvar
sb_lvar <- s_params$sb_lvar

z_mu <- 1/2 * (a_mu + b_mu + sa_mu + sb_mu)
z_var <- (exp(a_lvar) + exp(b_lvar) + exp(sa_lvar) + exp(sb_lvar)) / 4

z_mode <- exp(z_mu - z_var)
z_mean <- exp(z_mu + z_var/2)


