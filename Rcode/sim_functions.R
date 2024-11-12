##################################################
## Project:   simulation functions
## Date:      Jul 17, 2024
## Author:    Arnie Seong
##################################################

# DATA GENERATION ----
## sim_linear_data ----
sim_linear_data <- function(
  n_obs = 100,
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
  x <- torch_randn(n_obs, d_in)
  y <- x$matmul(true_coefs)$unsqueeze(2) + 
    intercept + 
    torch_normal(mean = 0, std = err_sigma, size = c(n_obs, 1))
  
  return(
    list(
      "y" = y,
      "x" = x,
      "true_coefs" = true_coefs,
      "intercept" = intercept,
      "n_obs" = n_obs,
      "d_in" = d_in,
      "d_true" = d_true
    )
  )
}


## sim_func_data----
fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x)
fcn3 <- function(x) abs(x)^(1.5)
fcn4 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x

sim_func_data <- function(
  n_obs = 1000,
  d_in = 10,
  flist = list(fcn1, fcn2, fcn3),
  err_sigma = 1
){
  # generate x, y
  x <- torch_randn(n_obs, d_in)
  y <- rep(0, n_obs)
  for(j in 1:length(flist)){
    y <- y + flist[[j]](x[,j])
  }
  y <- y$unsqueeze(2) + torch_normal(mean = 0, std = err_sigma, size = c(n_obs, 1))
  
  return(
    list(
      "y" = y,
      "x" = x,
      "n_obs" = n_obs,
      "d_in" = d_in,
      "d_true" = length(flist)
    )
  )
}




## sim_func_data_unifx----
sim_func_data_unifx <- function(
    n_obs = 1000,
    d_in = 10,
    flist = list(fcn1, fcn2, fcn3),
    err_sigma = 1,
    xlo = -5,
    xhi = 5
){
  # simulate covariates X from uniform distributions
  # generate x, y
  x <- (xlo - xhi) * torch_rand(n_obs, d_in) + xhi
  y <- rep(0, n_obs)
  for(j in 1:length(flist)){
    y <- y + flist[[j]](x[,j])
  }
  y <- y$unsqueeze(2) + torch_normal(mean = 0, std = err_sigma, size = c(n_obs, 1))
  
  return(
    list(
      "y" = y,
      "x" = x,
      "n_obs" = n_obs,
      "d_in" = d_in,
      "d_true" = length(flist)
    )
  )
}



# ASSESSMENT FUNCS ----

## binary error----
### binary_err_mat ----
binary_err_mat <- function(est, tru){
  # returns 4-row matrix of FP, TP, FN, TN
  FP <- est - tru == 1
  TP <- est + tru == 2
  FN <- est - tru == -1
  TN <- abs(tru) + abs(est) == 0
  return(rbind(FP, TP, FN, TN))
}

### binary_err----
binary_err <- function(est, tru){
  # returns FP, TP, FN, TN as percentage of all decisions
  rowSums(binary_err_mat(est, tru)) / length(tru)  
}

### binary_err_rate----
binary_err_rate <- function(est, tru){
  # returns FP, TP, FN, TN rates
  decision_counts <- rowSums(binary_err_mat(est, tru))
  actual_pos <- decision_counts[2] + decision_counts[3]
  actual_neg <- decision_counts[1] + decision_counts[4]
  
  denom <- c(actual_neg, actual_pos, actual_pos, actual_neg)
  # in case no positives or negatives predicted
  denom <- ifelse(denom == 0, 1, denom) 
  decision_counts / denom
}

## plotting predicted functions ----

### make_pred_mats----
make_pred_mats <- function(flist, xgrid = seq(-4.9, 5, length.out = 100), d_in){
  require(torch)
  n_truevars <- length(flist)
  # make torch array to pass into torchmod
  x_grids <- torch_zeros(n_truevars*length(xgrid), d_in)
  y_fcn <- rep(NA, n_truevars*length(xgrid))
  for(i in 1:n_truevars){
    st_row <- i*length(xgrid)-(length(xgrid)-1)
    end_row <- i*length(xgrid)
    x_grids[st_row:end_row, i] <- xgrid
    y_fcn[st_row:end_row] <- flist[[i]](xgrid)
  }
  
  vis_df <- data.frame(
    "y_true" = y_fcn,
    "x" = rep(xgrid, n_truevars),
    "name" = rep(paste0("x.", 1:n_truevars), each = length(xgrid))
  ) 
  
  return(
    list(
      "vis_df" = vis_df,
      "x_tensor" = x_grids
    )
  )
}

### plot_fcn_preds----
plot_fcn_preds <- function(torchmod, pred_mats, want_df = FALSE, want_plot = TRUE){
  vis_df <- pred_mats$vis_df
  vis_df$y_pred <- as_array(torchmod(pred_mats$x_tensor))
  plt <- vis_df %>% 
    ggplot() + 
    geom_line(
      aes(y = y_pred, x = x, color = name)
    ) + 
    geom_line(
      aes(
        y = y_true,
        x = x,
        color = name
      ),
      linewidth = 1,
      alpha = 0.15
    ) + 
    labs(
      title = "predicted and true fcns",
      color = ""
    )
  
  if (want_df & want_plot){
    return(list(
      "vis_df" = vis_df,
      "plt" = plt
    ))
  } else if (want_df) {
    return(vis_df)
  } else if (want_plot){
    plt
  }
}




# MISC UTILITIES ----
## time_since ----
time_since <- function(){
  # prints time since last called 
  # (use to print time a loop takes, for example).
  # usage:
  # f <- time_since()
  # f()
  # f()
  
  st <- Sys.time()
  function(x = Sys.time()) {
    print(x-st)
    st <<- x
  }
}


## cat_color(txt, style = 1, color = 36) ---
cat_color <- function(txt, style = 1, color = 36){
  # prints txt with colored font/bkgrnd
  cat(
    paste0(
      "\033[0;",
      style, ";",
      color, "m",
      txt,
      "\033[0m","\n"
    )
  )  
}


## mathematical operations
softplus <- function(x){
  log(1 + exp(x))
}

sigmoid <- function(x){
  1/(1 + exp(-x))
}

softmax <- function(z){
  exp(z) / sum(exp(z))
}

clamp <- function(v, lo = 0, hi = 1){
  #ensure values are in [0, 1]
  v <- ifelse( v <= hi, v, hi)
  v <- ifelse( v >= lo, v, lo)
  return(v)
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
  if (simdat$d_in > simdat$n_obs){
    lm_df <- lm_df[, 1:(ceiling(simdat$n_obs/2)+1)]
  }
  
  lm_fit <- lm(y ~ ., lm_df)
  if (length(simdat$true_coefs) >= simdat$n_obs){
    warning("p >= n; (p - n) + floor(n/2) spurious covariates eliminated to accomodate lm")
    calc_lm_stats(
      lm_fit = lm_fit, 
      true_coefs = simdat$true_coefs[1:ceiling(simdat$n_obs/2)], 
      alpha = alpha
    )
  } else {
    calc_lm_stats(lm_fit = lm_fit, true_coefs = simdat$true_coefs, alpha = alpha)
  }
}



# FOR SPIKE-SLAB ----



# FOR MOMBF ----


# FOR LASSO ----


