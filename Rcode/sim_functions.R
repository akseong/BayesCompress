##################################################
## Project:   simulation functions
## Date:      Jul 17, 2024
## Author:    Arnie Seong
##################################################

# DATA GENERATION ----
## sim_linear_data ----
sim_linear_data <- function(
  n = 100,
  d_in = 10,
  d_true = 3,
  err_sigma = 1,
  intercept = 0,
  true_coefs = NULL
){
  require(torch)
  
  # if true_coefs not provided, generates randomly
  if (is.null(true_coefs)){
    true_coefs <- round(runif(d_in,-5, 5), 2)
    true_coefs[(d_true + 1): d_in] <- 0
  }
  
  # ensure d_in, d_true match true_coefs (if true_coefs provided)
  d_in <- length(true_coefs)
  d_true <- sum(true_coefs != 0)
  
  # generate x, y
  x <- torch_randn(n, d_in)
  y <- x$matmul(true_coefs)$unsqueeze(2) + 
    intercept + 
    torch_normal(mean = 0, std = err_sigma, size = c(n, 1))
  
  return(
    list(
      "y" = y,
      "x" = x,
      "true_coefs" = true_coefs,
      "intercept" = intercept,
      "n" = n,
      "d_in" = d_in,
      "d_true" = d_true
    )
  )
}


## sim_func_data----
fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x)
fcn3 <- function(x) abs(x)^(1.5)

sim_func_data <- function(
  n = 1000,
  d_in = 10,
  flist = list(fcn1, fcn2, fcn3),
  err_sigma = 1
){
  # generate x, y
  x <- torch_randn(n, d_in)
  y <- rep(0, n)
  for(j in 1:length(flist)){
    y <- y + flist[[j]](x[,j])
  }
  y <- y$unsqueeze(2) + torch_normal(mean = 0, std = err_sigma, size = c(n, 1))
  
  return(
    list(
      "y" = y,
      "x" = x,
      "n" = n,
      "d_in" = d_in,
      "d_true" = length(flist)
    )
  )
}



# ASSESSMENT FUNCS ----
## binary_err_mat ----
binary_err_mat <- function(est, tru){
  # returns 4-row matrix of FP, TP, FN, TN
  FP <- est - tru == 1
  TP <- est + tru == 2
  FN <- est - tru == -1
  TN <- abs(tru) + abs(est) == 0
  return(rbind(FP, TP, FN, TN))
}

# binary_err----
binary_err <- function(est, tru){
  # returns FP, TP, FN, TN as percentage of all decisions
  rowSums(binary_err_mat(est, tru)) / length(tru)  
}

# binary_err_rate----
binary_err_rate <- function(est, tru){
  # returns FP, TP, FN, TN rates
  decision_counts <- rowSums(binary_err_mat(est, tru))
  pred_pos <- decision_counts[1] + decision_counts[2]
  pred_neg <- decision_counts[3] + decision_counts[4]
  
  denom <- c(pred_pos, pred_pos, pred_neg, pred_neg)
  # in case no positives or negatives predicted
  denom <- ifelse(denom == 0, 1, denom) 
  decision_counts / denom
}


# FOR LM() ----
##calc_lm_stats----
calc_lm_stats <- function(lm_fit, true_coefs, alpha = 0.05){
  beta_hat <- summary(lm_fit)$coef[-1, 1]
  binary_err <- binary_err_rate(
    est = summary(lm_fit)$coef[-1, 4] < alpha, 
    tru = true_coefs != 0)
  fit_mse <- mean(lm_fit$residuals^2)
  coef_mse <- mean((beta_hat - true_coefs)^2)
  list(
    "binary_err" = binary_err,
    "fit_mse" = fit_mse,
    "coef_mse" = coef_mse
  )
}

## get_lm_stats ----
get_lm_stats <- function(simdat, alpha = 0.05){
  lm_df <- data.frame(
    "y" = as_array(simdat$y), 
    "x" = as_array(simdat$x)
  )
  if (simdat$d_in > simdat$n){
    lm_df <- lm_df[, 1:(ceiling(simdat$n/2)+1)]
  }
  
  lm_fit <- lm(y ~ ., lm_df)
  if (length(simdat$true_coefs) >= n_obs){
    warning("p >= n; (p - n) + floor(n/2) spurious covariates eliminated to accomodate lm")
    calc_lm_stats(
      lm_fit = lm_fit, 
      true_coefs = simdat$true_coefs[1:ceiling(simdat$n/2)], 
      alpha = alpha
    )
  } else {
    calc_lm_stats(lm_fit = lm_fit, true_coefs = simdat$true_coefs, alpha = alpha)
  }
}



# FOR SPIKE-SLAB ----



# FOR MOMBF ----


# FOR LASSO ----



