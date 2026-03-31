##################################################
## Project:   get PIPs using kappa, alpha, recalibrated tau, downstream amplification
## Date:      Mar 30, 2026
## Author:    Arnie Seong
##################################################



frob_normalized <- function(mat) {
  frob(mat) / sqrt(min(nrow(mat), ncol(mat)))
}


get_lambda_j <- function(nn_layer, ln_fcn = ln_mode) {
  # lambda_j = sqrt(ztilde_sq), where ztilde_sq = atilde * btilde
  sqrt(get_ztil_sq(nn_layer, ln_fcn))
}

get_sigma_hat <- function(nn_mod, x, y, ln_fcn = ln_mode) {
  # Residual std from deterministic predictions
  # Requires the eval-mode forward pass or get_nn_mod_Ey
  y_pred <- get_nn_mod_Ey(nn_mod, x, ln_fcn)
  residuals <- as_array(y) - as_array(y_pred)
  sd(residuals)
}


recalibrate_tau <- function(lambda_j, sigma_hat, n, p, 
                            tau_range = c(1e-10, 1e3),
                            verbose = TRUE) {
  lam2 <- lambda_j^2
  
  equation <- function(log_tau) {
    tau  <- exp(log_tau)
    tau2 <- tau^2
    # Left side: effective number of features
    m_eff <- sum(tau2 * lam2 / (1 + tau2 * lam2))
    # Right side: implied p0 from P&V formula
    p0 <- p * tau * sqrt(n) / (sigma_hat + tau * sqrt(n))
    m_eff - p0
  }
  
  # Verify opposite signs at endpoints
  f_lo <- equation(log(tau_range[1]))
  f_hi <- equation(log(tau_range[2]))
  if (sign(f_lo) == sign(f_hi)) {
    warning(
      "No sign change on interval [", tau_range[1], ", ", tau_range[2], "]. ",
      "f(lo) = ", round(f_lo, 4), ", f(hi) = ", round(f_hi, 4), 
      ". Try widening tau_range."
    )
    return(NULL)
  }
  
  result <- uniroot(
    equation,
    interval = log(tau_range),
    tol = .Machine$double.eps^0.5
  )
  
  tau_star   <- exp(result$root)
  kappa_star <- 1 / (1 + tau_star^2 * lam2)
  m_eff      <- sum(1 - kappa_star)
  
  if (verbose) {
    cat("tau* =", round(tau_star, 6), "\n")
    cat("m_eff =", round(m_eff, 2), "effective features\n")
    cat("kappa range: [", round(min(kappa_star), 4), ",", 
        round(max(kappa_star), 4), "]\n")
  }
  
  list(
    "tau_star"   = tau_star,
    "kappa_star" = kappa_star,
    "m_eff"      = m_eff,
    "converged"  = result
  )
}




get_kappas_composed <- function(nn_mod, ln_fcn = ln_mean) {
  num_layers <- length(nn_mod$children)
  
  # Compose downstream matrix: M = W_L diag(z_L) ... W_2 diag(z_2)
  # Start from the output layer and work backward to layer 2
  if (!is.null(nn_mod$children[[num_layers]]$weight)){
    # if layer is a deterministic layer
    M <- as.matrix(nn_mod$children[[num_layers]]$weight)
  } else if (names(nn_mod$children[num_layers]) == "vc"){
    # if layer is a vc output layer
    M <- diag(sqrt(get_zsq(nn_mod$children[[num_layers]], ln_fcn)))
  } else {
    # if layer is a horseshoe layer
    M <- diag(sqrt(get_zsq(nn_mod$children[[num_layers]], ln_fcn)))
    M <- as.matrix(nn_mod$children[[num_layers]]$weight_mu) %*% M
  }
  
  if (num_layers > 2) {
    for (l in (num_layers - 1):2) {
      if (!is.null(nn_mod$children[[l]]$weight)){
        # if layer is a deterministic layer
        W_l <- as.matrix(nn_mod$children[[l]]$weight)
        M <- M %*% W_l
      } else if (names(nn_mod$children[l]) == "vc"){
        # if layer is a vc output layer
        z_l <- diag(sqrt(get_zsq(nn_mod$children[[l]], ln_fcn)))
        M <- M %*% z_l
      } else {
        # if layer is a horseshoe layer
        z_l <- diag(sqrt(get_zsq(nn_mod$children[[l]], ln_fcn)))
        W_l <- as.matrix(nn_mod$children[[l]]$weight_mu)
        M <- M %*% W_l %*% z_l
      }
    print(l)
    }
  }
  
  # M is now (d_out × d_hidden1)
  # Per-feature effective coefficient:
  #   beta_j^eff = z_{1,j} * || M %*% W_1[:,j] ||_2
  z1 <- sqrt(get_zsq(nn_mod$children[[1]], ln_fcn))
  W1 <- as.matrix(nn_mod$children[[1]]$weight_mu)
  
  # MW1 has shape (d_out × d_in); column j is M %*% W1[:,j]
  MW1 <- M %*% W1
  col_norms <- sqrt(colSums(MW1^2))  # ||M W1[:,j]||_2 for each j
  
  beta_eff <- z1 * col_norms
  kappas <- 1 / (1 + beta_eff^2)
  
  return(list(
    "kappas" = kappas,
    "beta_eff" = beta_eff,
    "M" = M,
    "col_norms" = col_norms
  ))
}



get_tau_corrected_spectral <- function(nn_mod, ln_fcn = ln_mean) {
  # Compose M as above, then use its spectral norm
  # ... (same composition code) ...
  spectral_norm <- max(svd(M)$d)
  
  tau_uncorrected <- sqrt(get_s_sq(nn_mod$children[[1]], ln_fcn))
  tau_corrected <- tau_uncorrected * spectral_norm
  return(tau_corrected)
}



get_beta_eff <- function(nn_mod, ln_fcn = ln_mean) {
  # not from gradient
  num_layers <- length(nn_mod$children)
  
  # Compose downstream: M = (W_L diag(z_L)) ... (W_2 diag(z_2))
  # Start from the output layer
  if (!is.null(nn_mod$children[[num_layers]]$atilde_mu)){
    z_L <- sqrt(get_zsq(nn_mod$children[[num_layers]], ln_fcn))
    if (names(nn_mod$children[num_layers]) != "vc"){
      W_L <- as_array(nn_mod$children[[num_layers]]$weight_mu)  
    } else {
      W_L = diag(1, length(z_L))
    }
    M <- W_L %*% diag(z_L)
  } else {
    M <- as_array(nn_mod$children[[num_layers]]$weight)
  }
  
  # Compose intermediate layers backward (L-1 down to 2)
  if (num_layers > 2) {
    for (l in (num_layers - 1):2) {
      if (!is.null(nn_mod$children[[l]]$weight_mu)){
        z_l <- sqrt(get_zsq(nn_mod$children[[l]], ln_fcn))
        W_l <- as_array(nn_mod$children[[l]]$weight_mu)
        M   <- M %*% W_l %*% diag(z_l)
      } else {
        W_l <- as_array(nn_mod$children[[l]]$weight)
        M   <- M %*% W_l
      }
      print(l)
    }
  }
  # M is now (d_out x d_hidden1)
  
  # First layer
  z_1 <- sqrt(get_zsq(nn_mod$children[[1]], ln_fcn))
  W_1 <- as_array(nn_mod$children[[1]]$weight_mu)
  
  # MW1 = M %*% W_1: (d_out x d_in)
  MW1 <- M %*% W_1
  
  # Per-feature downstream amplification
  c_j <- sqrt(colSums(MW1^2))    # ||M W_1[,j]||_2 for each j
  
  # Effective coefficient
  beta_eff <- z_1 * c_j
  
  list(
    "beta_eff" = beta_eff,    # full effective coefficient per feature
    "c_j"      = c_j,         # downstream amplification only (for get_posterior_inclusion_prob)
    "z_1"      = z_1,         # first-layer gating values
    "M"        = M,           # composed downstream matrix
    "MW1"      = MW1          # full composition before z_1 gating
  )
}


get_beta_eff_gradient <- function(nn_mod, x_data, n_mc = 20) {
  # for use without eval() implementation
  d_in <- x_data$shape[2]
  grad_accum <- rep(0, d_in)
  
  for (s in seq_len(n_mc)) {
    x_t <- x_data$detach()$clone()$requires_grad_(TRUE)
    y_pred <- nn_mod(x_t)
    y_pred$sum()$backward()
    grad_accum <- grad_accum + as_array(x_t$grad$abs()$mean(dim = 1))
  }
  
  grad_accum / n_mc
}


get_beta_eff_gradient_det <- function(nn_mod, x_data) {
  # for use with eval() implementation in forward pass
  nn_mod$eval()
  x_t <- x_data$detach()$clone()$requires_grad_(TRUE)
  y_pred <- nn_mod(x_t)
  y_pred$sum()$backward()
  beta_eff <- as_array(x_t$grad$abs()$mean(dim = 1))
  nn_mod$train()
  beta_eff
}


get_posterior_inclusion_prob <- function(
    nn_layer, 
    tau_star = NULL,        # recalibrated tau; NULL = use learned global
    c_j = NULL,             # downstream correction; NULL = no correction
    threshold = 0.5         # kappa threshold (0.5 = more signal than noise)
) {
  # Local parameters
  mu_local <- (as_array(nn_layer$atilde_mu) + 
                 as_array(nn_layer$btilde_mu)) / 2
  var_local <- (exp(as_array(nn_layer$atilde_logvar)) + 
                  exp(as_array(nn_layer$btilde_logvar))) / 4
  
  # Global component
  if (!is.null(tau_star)) {
    mu_global <- log(tau_star)
    var_global <- 0  # fixed after recalibration
  } else {
    mu_global <- (as_array(nn_layer$sa_mu) + 
                    as_array(nn_layer$sb_mu)) / 2
    var_global <- (exp(as_array(nn_layer$sa_logvar)) + 
                     exp(as_array(nn_layer$sb_logvar))) / 4
  }
  
  # Downstream correction
  log_c <- if (!is.null(c_j)) log(c_j) else 0
  
  # Combined parameters for log(z_j)
  mu_z  <- mu_local + mu_global + log_c
  sig_z <- sqrt(var_local + var_global)
  
  # P(kappa_j < threshold)
  log_odds <- 0.5 * log((1 - threshold) / threshold)  # 0 when threshold = 0.5
  P_j <- pnorm((mu_z - log_odds) / sig_z)
  
  # Dropout probability for BFDR
  dropout_prob <- 1 - P_j
  
  # Signal-to-noise ratio (for diagnostics)
  snr <- mu_z / sig_z
  
  return(list(
    "P_j" = P_j,
    "dropout_prob" = dropout_prob,
    "snr" = snr,
    "mu_z" = mu_z,
    "sig_z" = sig_z
  ))
}


# effective betas code ----

## --- Composed matrix approach ---
model_fit <- nn_mod
beff <- get_beta_eff(model_fit, ln_fcn = ln_mode)
round(beff$beta_eff, 3)
kappas_local <- get_kappas(model_fit$fc1, type = "local")
alphas <- as_array(model_fit$fc1$get_dropout_rates())






# Feed into recalibrate_tau as pseudo-linear coefficients

sigma_hat <- get_sigma_hat(model_fit, x_train, y_train, ln_fcn = ln_mean)
lambda_eff <- beff$beta_eff / median(beff$beta_eff)  # normalize to relative scales
rc <- recalibrate_tau(lambda_eff, sigma_hat, n = ttsplit_ind, p = sim_res$sim_params$d_in)

# Or feed c_j into posterior inclusion probability
pip <- get_posterior_inclusion_prob(
  model_fit$fc1,
  tau_star = rc$tau_star,
  c_j      = beff$c_j
)

# Variable selection
err_from_dropout(
  dropout_vec = pip$dropout_prob,
  max_bfdr    = 0.05,
  true_gam    = c(rep(1, 4), rep(0, 100))
)


## --- Gradient-based cross-check ---
beff_grad <- get_beta_eff_gradient(model_fit, x_train, n_mc = 20)

plot_beta_comparison <- function(beff_composed, beff_gradient, d_true = 4) {
  p <- length(beff_composed)
  feature_type <- c(rep("signal", d_true), rep("noise", p - d_true))
  
  df <- data.frame(
    feature   = 1:p,
    composed  = beff_composed,
    gradient  = beff_gradient,
    type      = feature_type
  )
  
  # Rank comparison
  p1 <- ggplot(df, aes(x = log(composed + 1e-10), 
                       y = log(gradient + 1e-10), 
                       color = type)) +
    geom_point(alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.4) +
    labs(title = "Composed vs gradient-based beta_eff (log scale)",
         x = "log(beta_eff composed)", 
         y = "log(beta_eff gradient)")
  
  # Side-by-side magnitudes
  p2 <- df %>%
    pivot_longer(cols = c(composed, gradient), 
                 names_to = "method", values_to = "beta") %>%
    ggplot(aes(x = feature, y = log(beta + 1e-10), 
               color = type, shape = method)) +
    geom_point(alpha = 0.7) +
    labs(title = "Feature importance by method (log scale)",
         y = "log(beta_eff)")
  
  gridExtra::grid.arrange(p1, p2, nrow = 2)
}

plot_beta_comparison(beff$beta_eff, beff_grad, d_true = 4)







# recalibrations for tau ----
# 1. Compute per-feature effective coefficients
model_fit <- nn_mod
result <- get_kappas_composed(model_fit, ln_fcn = ln_mean)
beta_eff <- result$beta_eff

# 2. Estimate residual sigma
y_pred <- get_nn_mod_Ey(model_fit, x_train, ln_fcn = ln_mean)
sigma_hat <- sd(as_array(y_train) - as_array(y_pred))

# 3. Recalibrate: treat beta_eff as pseudo-linear coefficients,
#    find self-consistent tau* using the P&V framework
lambda_eff <- beta_eff / median(beta_eff)  # normalize to get relative scales
tau_star <- recalibrate_tau(lambda_eff, sigma_hat, sim_res$sim_params$n_obs, p = 104)$tau_star


# 4. Compute corrected kappas
kappa_star <- 1 / (1 + tau_star^2 * lambda_eff^2)



# PIPs ----
# Usage with existing BFDR infrastructure:
pip <- get_posterior_inclusion_prob(
  model_fit$fc1, 
  tau_star = tau_star,    # from self-consistency recalibration
  c_j = col_norms         # from composed downstream matrix
)

# Plug directly into existing functions:
err_from_dropout(
  dropout_vec = pip$dropout_prob,
  max_bfdr = 0.05,
  true_gam = c(rep(1, 4), rep(0, 100))
)


get_posterior_inclusion_prob(model_fit$fc1, threshold = 0.75)
























# my code ----

n = 1e4
sig_hat <- 1.2
recalibrate_tau <- function(nn_mod, n, sig_hat){
  # just wants to set tau at 0.
  p <- length(nn_mod$fc1$atilde_mu)
  lambdas <- sqrt(get_ztil_sq(nn_mod$fc1))
  
  m_eff_to_opt <- function(tau){
    numerator <- (tau^2 * lambdas^2)
    sum( numerator / (1 + numerator))
  }
  
  exp_m_to_opt <- function(tau){
    p*tau*sqrt(n) / (sig_hat + tau*sqrt(n))
  }
  
  opt_fcn <- function(tau){
    abs(exp_m_to_opt(tau) - m_eff_to_opt(tau))
  }
  
  t1 <- optim(
    par = sim_res$sim_params$prior_tau,
    fn = opt_fcn,
    method = "Brent",
    lower = 0,
    upper = .01
  )$par
  
}


kappas_local


# my code for pips ----
ztil_params <- get_ztil_params(nn_mod$fc1)
ztil_mu <- (ztil_params$at + ztil_params$bt)/2
ztil_var <- (exp(ztil_params$at_lvar) + exp(ztil_params$bt_lvar))/4

s_params <- get_s_params(nn_mod$fc1)
s_mu <- (s_params$sa + s_params$sb)/2
s_var <- (exp(s_params$sa_lvar) + exp(s_params$sb_lvar))/4

ztil_mu/sqrt(ztil_var)


kappas_local <- get_kappas(nn_mod$fc1, type = "local")
alphas <- as_array(nn_mod$fc1$get_dropout_rates())
sqrt(alphas)

# tau correction from layer 2
m2 <- m_eff(nn_mod$fc2)
d1 <- sim_res$sim_params$d_hidden1




pnorm(sqrt(alphas))
pnorm((ztil_mu + s_mu) / sqrt(log(1 + alphas)))

tfcn <- function(thresh)(log((1-thresh)/thresh))
pnorm((ztil_mu + s_mu + log(d1/m2)/2 - 0.5*tfcn(0.05)) / sqrt(log(1 + alphas)))

pnorm((ztil_mu + s_mu + log(spectral_norm) - 0.5*tfcn(0.05)) / sqrt(log(1 + alphas)))



get_pips <- function(nn_mod, thresh = 0.05, correction_type = "m_eff"){
  # calculates posterior probability that kappas < thresh
  ztil_params <- get_ztil_params(nn_mod$fc1)
  ztil_mu <- (ztil_params$at + ztil_params$bt)/2
  ztil_var <- (exp(ztil_params$at_lvar) + exp(ztil_params$bt_lvar))/4
  
  s_params <- get_s_params(nn_mod$fc1)
  s_mu <- (s_params$sa + s_params$sb)/2
  s_var <- (exp(s_params$sa_lvar) + exp(s_params$sb_lvar))/4
  z_var <- ztil_var + s_var
  
  correction = 0
  if (correction_type == "m_eff"){
    # tau correction from layer 2
    correction = log(tau_correction(nn_mod))
  }
  
  alphas <- as_array(nn_mod$fc1$get_dropout_rates(type = "marginal"))
  
  pnorm((ztil_mu + s_mu + correction - log((1-thresh)/thresh)/2)/ sqrt(z_var))
}

# calibrate bfdr based on kappa threshold
kappa_thresh <- 0.05
get_kappas(nn_mod$fc1, type = "local")
get_kappas_taucorrected(nn_mod)
dropout_probs <- 1 - get_pips(nn_mod, thresh = kappa_thresh, correction = "m_eff")
true_gam <- c(rep(1, 4), rep(0, 100))

BFDR_eta_search(dropout_probs, max_rate = 0.05)
BFDR(dropout_probs, BFDR_eta_search(dropout_probs, max_rate = 0.05))

kappas_tau <- get_kappas_taucorrected(nn_mod)
kappas_composed <- get_kappas_composed(nn_mod)
BFDR(kappas_tau, BFDR_eta_search(kappas_tau, max_rate = 0.05))
round(kappas_tau, 2)


sim_res$sim_params$n_obs



ztil_mu + s_mu + log(m2)



m2 <- m_eff(nn_mod$fc2)



