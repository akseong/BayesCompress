##################################################
## Project:   simulation functions
## Date:      Jul 17, 2024
## Author:    Arnie Seong
##################################################


# MISC UTILITIES ----

`%notin%` <- Negate(`%in%`)

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

find_array_ind <- function(array, name, marg = 1){
  # marg = 1 --> rows;    marg = 2 --> cols
  # generalizes to higher-dim arrays
  inds <- which(dimnames(array)[[marg]] == name)
  
  if (length(inds) == 0){
    warning("no corresponding row found")
  } else if (length(inds) > 1) {
    warning("multiple corresponding rows found")
    return(inds)
  } else if (length(inds) == 1) {
    return(inds)
  }
}

update_matrix_row <- function(mat, epoch, update_vec, reportevery = 100, verbose = FALSE){
  # USAGE EXAMPLE:  
  #    mat <- matrix(NA, nrow = 20, update_vec = rep(1, 5) ncol = 5)
  #    for (i in 1:1000){
  #      mat <- matfunc(mat, epoch = i)
  #    }
  
  if (!epoch%%reportevery){
    row_ind <- epoch%/%reportevery
    mat[row_ind, ] <- update_vec
    if (verbose) {
      cat("row", row_ind, 
          "updated on epoch", epoch, 
          "with", update_vec,
          sep = " ")
    }
  } else if (epoch%%reportevery & verbose){
    cat("no update on epoch ", epoch)
  }
  return(mat)
}

cat_color <- function(txt, sep_char = ", ", style = 1, color = 36){
  # prints txt with colored font/bkgrnd
  cat(
    paste0(
      "\033[0;",
      style, ";",
      color, "m",
      txt,
      "\033[0m"
    ),
    sep = sep_char
  )  
}

append_txt_file <- function(file_path, msg){
  # used to keep track of simulation progress on server
  fileConn<-file(file_path, "a") # "a" specifies "append"
  writeLines(msg, fileConn)
  close(fileConn)
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

ln_mode <- function(mu, var){
  # log-normal distribution mode
  exp(mu - var)
}

ln_mean <- function(mu, var){
  # log-normal distribution expected value
  exp(mu + var/2)
}

geom_mean <- function(vec){
  # geometric mean
  exp(mean(log(vec)))
}

vismat <- function(mat, cap = NULL, lims = NULL, leg = TRUE, na0 = TRUE, square){
  # outputs visualization of matrix with few unique values
  # colnames should be strings, values represented as factors
  # sci_not=TRUE puts legend in scientific notation
  require(ggplot2)
  require(scales)
  require(reshape2)
  
  melted <- melt(mat)
  melted$value <- ifelse(
    melted$value == 0 & na0,
    NA,
    melted$value
  )
  p <- ggplot(melted) + 
    geom_raster(aes(y = Var1, 
                    x = Var2, 
                    fill = value)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    scale_fill_viridis_c(limits = lims)
  
  if (is.numeric(melted$Var1)){
    p <- p + 
      scale_y_reverse()
  } else {
    p <- p + 
      scale_y_discrete(limits = rev(levels(melted$Var1)))
  }
  
  if (missing(square)) square <- nrow(mat) / ncol(mat) > .9 & nrow(mat) / ncol(mat) < 1.1
  if (square) p <- p + coord_fixed(1)
  
  if (!is.null(cap)) p <- p + labs(title=cap)
  
  if (!leg) p <- p + theme(legend.position = "none")
  
  return(p)
}

roll_vec <- function(vec, new_vals){
  # use to store needed values across epochs.  
  # moves window forward by however many values added
  vec <- c(vec[(length(new_vals) + 1): length(vec)], new_vals)
  return(vec)
}
# example code:
# > test_mse_store <- rep(NA, 102)
# > stopcrit <- FALSE
# > i = 1
# > while (i < 100000 & !stopcrit){
# >   te_mse <- rnorm(1)
# >   test_mse_store <- roll_vec(test_mse_store, new_vals = te_mse)
# >   if (i > 102){
# >     stopcrit <- abs(.001 * sd(test_mse_store[1:100])) > abs(diff(test_mse_store[101:102]))
# >   }  
# >   i = i+1
# > }

# CUDA ----

dev_select <- function(use_cuda){
  ifelse(use_cuda, "cuda", "cpu")
}

dev_auto <- function(){
  ifelse(torch::cuda_is_available(), "cuda", "cpu")
}


# DATA GENERATION ----
plot_datagen_fcns <- function(
    flist, 
    min_x = -3, 
    max_x = 3, 
    x_length = 100
  ){
  xshow <- seq(min_x, max_x, length.out = x_length)
  yshow <- sapply(flist, function(fcn) fcn(xshow))
  colnames(yshow) <- paste0("f", 1:length(flist))
  df <- data.frame(cbind(yshow, "x" = xshow))
  plt <- df %>% 
    pivot_longer(cols = -x, names_to = "fcn") %>%
    ggplot(aes(y = value, x = x, color = fcn)) +
    geom_line() +
    labs(title = "functions used to create data")

  return(plt)
}

sim_linear_data <- function(
  n_obs = 100,
  err_sigma = 1,
  true_coefs = NULL,
  d_in = 10,
  d_true = 3,
  intercept = 0
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

# # sample functions to use
# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x)
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x

# # slightly harder setting
# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) - (abs(x))
# flist = list(fcn1, fcn2, fcn3, fcn4)

sim_func_data <- function(
  n_obs = 1000,
  d_in = 10,
  flist = list(fcn1, fcn2, fcn3, fcn4),
  err_sigma = 1,
  use_cuda = FALSE
){
  # generate x, y
  x <- torch_randn(n_obs, d_in)
  y <- rep(0, n_obs)
  for(j in 1:length(flist)){
    y <- y + flist[[j]](x[,j])
  }
  y <- y$unsqueeze(2) + torch_normal(mean = 0, std = err_sigma, size = c(n_obs, 1))
  
  if (use_cuda){
    x <- x$to(device = "cuda")
    y <- y$to(device = "cuda")
  }
  
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


sim_func_data_unifx <- function(
    n_obs = 1000,
    d_in = 10,
    flist = list(fcn1, fcn2, fcn3, fcn4),
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
binary_err_mat <- function(est, tru){
  # returns 4-row matrix of FP, TP, FN, TN
  FP <- est - tru == 1
  TP <- est + tru == 2
  FN <- est - tru == -1
  TN <- abs(tru) + abs(est) == 0
  return(rbind(FP, TP, FN, TN))
}

binary_err <- function(est, tru){
  # returns FP, TP, FN, TN as percentage of all decisions
  rowSums(binary_err_mat(est, tru)) / length(tru)  
}

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

BFDR <- function(dropout_probs, eta){
  delta_vec <- 1 * (dropout_probs <= eta)
  if (sum(delta_vec) == 0){ 
    warning("no included variables; returning BFDR = 0") 
    bfdr <- 0
  } else {
    bfdr <- sum((dropout_probs) * delta_vec) / sum(delta_vec)
  }
  return(
    list(
      "delta_i" = delta_vec,
      "bfdr" = bfdr
    )
  )
}

FDR <- function(delta_vec, true_gam = c(rep(1, 4), rep(0, 100))){
  if (sum(delta_vec) == 0){
    return(0)
  } else {
    return(sum((delta_vec - true_gam) == 1) / sum(delta_vec))
  }
}

BFDR_eta_search <- function(dropout_probs, max_rate = 0.05){
  a_sort <- sort(dropout_probs)
  bfdrs <- sapply(a_sort, function(X) BFDR(dropout_probs, eta = X)$bfdr)
  inds <- which(bfdrs <= max_rate)
  if (length(inds)==0){
    warning("no threshold found, returning eta = 0")
    return(0)
  } else {
    a_sort[max(inds)]
  }
}

err_from_dropout <- function(
    dropout_vec, 
    max_bfdr = 0.01, 
    true_gam = c(rep(1, 4), rep(0, 100))
){
  eta <- BFDR_eta_search(dropout_vec, max_rate = max_bfdr)
  bfdr <- BFDR(dropout_vec, eta)$bfdr
  delta_i <- BFDR(dropout_vec, eta)$delta_i
  bin_err <- binary_err_rate(est = delta_i, tru = true_gam)
  fdr <- FDR(delta_vec = delta_i, true_gam = true_gam)
  c("fdr" = fdr, "bfdr" = bfdr, bin_err)
}

get_s_params <- function(nn_model_layer){
  sa <- as_array(nn_model_layer$sa_mu)
  sb <- as_array(nn_model_layer$sb_mu)
  sa_lvar <- as_array(nn_model_layer$sa_logvar)
  sb_lvar <- as_array(nn_model_layer$sb_logvar)
  return(
    list(
      "sa" = sa,
      "sb" = sb,
      "sa_lvar" = sa_lvar,
      "sb_lvar" = sb_lvar
    )
  )
}

get_ztil_params <- function(nn_model_layer){
  atil <- as_array(nn_model_layer$atilde_mu)
  btil <- as_array(nn_model_layer$btilde_mu)
  atil_lvar <- as_array(nn_model_layer$atilde_logvar)
  btil_lvar <- as_array(nn_model_layer$btilde_logvar)
  return(
    list(
      "at" = atil,
      "bt" = btil,
      "at_lvar" = atil_lvar,
      "bt_lvar" = btil_lvar
    )
  )
}

get_s_sq <- function(nn_model_layer, ln_fcn = ln_mode){
  s_params <- get_s_params(nn_model_layer)
  s_sq <- ln_fcn(
    s_params$sa + s_params$sb, 
    exp(s_params$sa_lvar) + exp(s_params$sb_lvar)
  )
  return(s_sq)
}

get_ztil_sq <- function(nn_model_layer, ln_fcn = ln_mode){
  ztil_params <- get_ztil_params(nn_model_layer)
  ztil_sq <- ln_fcn(
    ztil_params$at + ztil_params$bt, 
    exp(ztil_params$at_lvar) + exp(ztil_params$bt_lvar)
  )
  return(ztil_sq)
}

get_kappas <- function(nn_model_layer, type = "global"){
  ztil_sq <- get_ztil_sq(nn_model_layer)
  if (type == "global"){
    s_sq <- get_s_sq(nn_model_layer)
    kappas <- 1 / ( 1 + s_sq*ztil_sq)
  } else if (type == "local"){
    kappas <- 1 / ( 1 + ztil_sq)
  } else {
    warning("type must be global or local")
  }
  return(kappas)
}

get_wtil_params <- function(nn_model_layer){
  wtil_lvar <- as_array(nn_model_layer$weight_logvar)
  wtil_mu <- as_array(nn_model_layer$weight_mu)
  return(
    list(
      "wtil_lvar" = wtil_lvar,
      "wtil_mu" = wtil_mu
    )
  )
}


get_Wz_params <- function(nn_model_layer){
  # these are the params for the CONDITIONAL W | z 
  wtil_params <- get_wtil_params(nn_model_layer)
  z_sq <- get_s_sq(nn_model_layer) * get_ztil_sq(nn_model_layer)
  # # checking sweep function
  # # want to multiply test_mat column j by element j in mult_vec
  # test_mat <- cbind(
  #   c(0,0,0,0,0),
  #   c(1, 1, 1, 1, 1),
  #   c(-1, -1, -1, -1, -1)
  # )
  # test_mat
  # mult_vec <- 1:3
  # sweep(test_mat, 2, STATS = mult_vec, FUN = "*")
  Wz_mu <- sweep(
    wtil_params$wtil_mu, 
    MARGIN = 2, 
    STATS = sqrt(z_sq), 
    FUN = "*"
  )
  Wz_var <- sweep(
    exp(wtil_params$wtil_lvar), 
    MARGIN = 2, 
    STATS = z_sq, 
    FUN = "*"
  )
  return(
    list(
      "Wz_mu" = Wz_mu,
      "Wz_var" = Wz_var
    )
  )
}


## plotting predicted functions ----
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




# FOR LM() ----
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


# SIMULATION FCNS ----
hot_start_DNN <- function(
    sim_ind,
    sim_params,
    verbose = TRUE,
    want_fcn_plots = TRUE,
    save_fcn_plots = FALSE,
    save_mod = TRUE,
    save_path_stem = NULL
){
  # train regular DNN for preliminary weights to save training time
  # returns trained DNN; can also save trained DNN
  if (save_mod) {
    if (is.null(save_path_stem)){
      save_path_stem <- paste0(
        "dnn_n", sim_params$n_obs, "_seed", 
        sim_params$sim_seeds[sim_ind]
      )
    }
    mod_save_path <- paste0(save_path_stem, ".pt")
  }
  
  # simulate same dataset via seed----
  set.seed(sim_params$sim_seeds[sim_ind])
  torch_manual_seed(sim_params$sim_seeds[sim_ind])
  
  simdat <- sim_func_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    err_sigma = sim_params$err_sig,
    use_cuda = sim_params$use_cuda
  )
  
  ttsplit_ind <- floor(sim_params$n_obs * sim_params$ttsplit)
  dnn_x <- simdat$x[1:ttsplit_ind, ] 
  dnn_y <- simdat$y[1:ttsplit_ind]
  
  # regular DNN to get starting weights ----
  DNN <- nn_sequential(
    nn_linear(sim_params$d_in, sim_params$d_hidden1),
    nn_relu(),
    nn_linear(sim_params$d_hidden1, sim_params$d_hidden2),
    nn_relu(),
    nn_linear(sim_params$d_hidden2, sim_params$d_out)
  )
  if (sim_params$use_cuda) {DNN$to(device = "cuda")}
  
  if (want_fcn_plots){
    pred_mats <- make_pred_mats(
      flist = sim_params$flist,
      xgrid = seq(-3, 3, length.out = 100),
      d_in = simdat$d_in
    )
  }
  
  learning_rate <- 0.08
  optimizer <- optim_adam(DNN$parameters, lr = learning_rate)
  
  for (t in 1:sim_params$hot_start_epochs) {
    y_pred <- DNN(dnn_x)
    loss <- nnf_mse_loss(y_pred, dnn_y)
    
    if (verbose & (t %% sim_params$report_every == 0)) {
      cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
      
      # plot true and pred functions
      if (want_fcn_plots){
        plt <- plot_fcn_preds(
          torchmod = DNN,
          pred_mats = pred_mats
        ) + 
          labs(
            title = "predicted (solid) and original (dotted) functions",
            subtitle = paste0("epoch: ", t)
          )
        
        if (save_fcn_plots){
          plt_save_path <- paste0(save_path_stem, "_e", t, ".png")
          ggsave(filename = plt_save_path, plot = plt)
        } else {
          print(plt)
        }
        
      }
    } # END UPDATE LOOP
    
    # gradient step
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()
  } # END TRAINING LOOP
  
  message("DNN finished training")
  
  if (save_mod){
    torch_save(model_fit, path = save_path)
    message(paste0("hot start model saved: ", save_path))
  }
  
  return(DNN)
}


sim_fcn_hshoe_linreg <- function(
    sim_ind,    # to ease parallelization
    sim_params,     # see example
    nn_model,   # torch nn_module,
    train_epochs = 10000,
    verbose = TRUE,      # provide updates in console
    report_every = 1000, # training epochs between display/store results
    want_plots = TRUE,   # provide graphical updates of KL, MSE
    want_all_params = FALSE, 
    want_data = FALSE    # include training data in output
){
  # - fcn used with lapply for parallel processing
  # - basic linear regression setting
  # - only stop criteria used is max epochs
  # - if want_all_params = FALSE, only returns
  #     loss_mat (KL, mse vs epoch) and
  #     alpha_mat (local dropout rates vs epoch)
  # USAGE EXAMPLE:
  # > sim_params <- list(
  # >   "sim_name" = "horseshoe, linear regression setting, KL scaled by n",
  # >   "n_sims" = 100,
  # >   "d_in" = 104,
  # >   "n_obs" = 125,
  # >   "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  # >   "seed" = 314,
  # >   "err_sig" = 1,
  # >   "ttsplit" = 4/5,
  # >   "stop_criteria" = c("test_train","ma_loss_increasing")
  # > )
  # >
  # > res <- lapply(
  # >   1:sim_params$n_sims, 
  # >   function(X) sim_fcn_hshoe_linreg(
  # >     sim_ind = X, 
  # >     sim_params = sim_params,
  # >     nn_model = SLHS,  #### this needs to be pre-defined via torch::nn_module()
  # >     train_epochs = 1000,
  # >     verbose = TRUE,
  # >     report_every = 100,
  # >     want_plots = TRUE,
  # >     want_all_params = FALSE,
  # >     want_data = FALSE
  # >   )
  # > )
  
  ## generate linear data
  set.seed(sim_params$sim_seeds[sim_ind])
  torch_manual_seed(sim_params$sim_seeds[sim_ind])
  
  lin_simdat <- sim_linear_data(
    n = sim_params$n_obs,
    true_coefs = sim_params$true_coefs,
    err_sigma = sim_params$err_sig
  )
  
  # store: # train, test mse and kl
  report_epochs <- seq(
    report_every, 
    train_epochs, 
    by = report_every
  )
  
  loss_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = 3
  )
  colnames(loss_mat) <- c("kl", "mse_train", "mse_test")
  rownames(loss_mat) <- report_epochs
  
  
  # store: alphas
  alpha_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = sim_params$d_in
  )
  rownames(alpha_mat) <- report_epochs
  
  # store: weight posterior params
  if (want_all_params){
    w_mu_mat <-
      w_var_mat <- 
      marginal_dropout_mat <-
      atilde_mu_mat <- 
      btilde_mu_mat <-
      atilde_logvar_mat <-
      btilde_logvar_mat <- alpha_mat
    
    global_dropout_vec <-   
      # tau_vec <- 
      sa_mu_vec <-
      sb_mu_vec <- 
      sa_logvar_vec <- 
      sb_logvar_vec <- rep(NA, length(report_epochs))
  }
  
  
  ## SETUP STOP CRITERIA ----
  #### 
  ####  NOT FULLY IMPLEMENTED IN THIS FILE.  ONLY FOR STORAGE / VIEWING. 
  ####  ONLY STOPPING CRITERIA HERE IS MAX_EPOCHS.
  ####
  ## test-train split
  
  ttsplit_used <- "test_train" %in% sim_params$stop_criteria || "test_convergence" %in% sim_params$stop_criteria
  ttsplit_ind <- ifelse(
    ttsplit_used,
    floor(sim_params$n_obs * sim_params$ttsplit),
    sim_params$n_obs
  )
  
  if (ttsplit_used){
    x_test <- lin_simdat$x[(ttsplit_ind+1):sim_params$n_obs, ] 
    y_test <- lin_simdat$y[(ttsplit_ind+1):sim_params$n_obs, ]
    loss_test <- torch_zeros(1)  # set initial value
    loss_diff_test <- 1          # set initial value    
  }
  
  x_train <- lin_simdat$x[1:ttsplit_ind, ]
  y_train <- lin_simdat$y[1:ttsplit_ind, ]
  
  ## initialize BNN & optimizer ----
  model_fit <- nn_model()
  optim_slhs <- optim_adam(model_fit$parameters)
  
  ## TRAINING LOOP ----
  ## initialize training params
  loss_diff <- 1
  loss <- torch_zeros(1)
  epoch <- 1
  stop_criteria_met <- FALSE
  
  while (!stop_criteria_met){
    prev_loss <- loss
    
    # fit & metrics
    yhat_train <- model_fit(x_train)
    mse <- nnf_mse_loss(yhat_train, y_train)
    kl <- model_fit$get_model_kld() / ttsplit_ind
    loss <- mse + kl
    # loss_diff <- loss - prev_loss
    
    
    # gradient step 
    # zero out previous gradients
    optim_slhs$zero_grad()
    # backprop
    loss$backward()
    # update weights
    optim_slhs$step()
    
    
    # compute test loss 
    if (ttsplit_used) {
      prev_loss_test <- loss_test
      yhat_test <- model_fit(x_test) 
      mse_test <- nnf_mse_loss(yhat_test, y_test)
    }
    
    
    ### store results (every `report_every` epochs) ----
    time_to_report <- epoch!=0 & (epoch %% report_every == 0)
    if (time_to_report){
      row_ind <- epoch %/% report_every
      
      loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_test$item())
      dropout_alphas <- model_fit$fc1$get_dropout_rates()
      alpha_mat[row_ind, ] <- as_array(dropout_alphas)
      
      # storing other optional parameters, mostly for diagnostics
      if (want_all_params){
        global_dropout_vec[row_ind] <- as_array(model_fit$fc1$get_dropout_rates(type = "global"))
        marginal_dropout_mat[row_ind, ] <- as_array(model_fit$fc1$get_dropout_rates(type = "marginal"))
        # tau_vec[row_ind] <- as_array(model_fit$fc1$tau)
        w_mu_mat[row_ind, ] <- as_array(model_fit$fc1$compute_posterior_param()$post_weight_mu)
        w_var_mat[row_ind, ] <- as_array(model_fit$fc1$compute_posterior_param()$post_weight_var)
        
        atilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_mu)
        btilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_mu)
        atilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_logvar)
        btilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_logvar)
        sa_mu_vec[row_ind] <- as_array(model_fit$fc1$sa_mu)
        sa_logvar_vec[row_ind] <- as_array(model_fit$fc1$sa_logvar)
        sb_mu_vec[row_ind] <- as_array(model_fit$fc1$sb_mu)
        sb_logvar_vec[row_ind] <- as_array(model_fit$fc1$sb_logvar)
      }
    } # end result storing and updating
    
    
    ## in-console and graphical training updates ----
    if (time_to_report & verbose){
      cat(
        "Epoch:", epoch,
        "MSE + KL/n =", mse$item(), "+", kl$item(),
        "=", loss$item(),
        "\n",
        "train mse:", round(mse$item(), 4), 
        "; test_mse:", round(mse_test$item(), 4),
        "\n", 
        sep = " "
      )
      cat("alphas: ", round(as_array(dropout_alphas), 2), "\n")
      display_alphas <- ifelse(
        as_array(dropout_alphas) <= .1,
        round(as_array(dropout_alphas), 3),
        "."
      )
      cat("alphas below 0.1: ")
      cat_color(display_alphas, sep = " ")
      cat(" \n \n")
      
      
      # graphical training updates
      if (want_plots & row_ind > 5){
        
        # only show most recent (for scale of plot)
        start_plot_row_ind <- 1
        if (row_ind >= 20){
          start_plot_row_ind <- row_ind - 10
        } else if (row_ind >= 60){
          start_plot_row_ind <- row_ind - 30
        }
        # set up plotting df
        temp_lmat <- cbind(
          loss_mat[start_plot_row_ind:row_ind,],
          "epoch" = report_epochs[start_plot_row_ind:row_ind]
        )
        
        mse_plotdf <- temp_lmat %>% 
          data.frame() %>% 
          pivot_longer(
            cols = -epoch,
            names_to = "metric"
          )
        
        # plot MSE vs epochs
        mse_plt <- mse_plotdf %>% 
          filter(metric != "kl") %>% 
          ggplot(aes(y = value, x = epoch, color = metric)) + 
          geom_line() + 
          labs(
            subtitle = paste0("MSE for epochs ", 
                              report_epochs[start_plot_row_ind],
                              " through ",
                              report_epochs[row_ind])
          )
        
        # plot KL vs epochs
        kl_plt <- mse_plotdf %>%
          filter(metric == "kl") %>%
          ggplot(aes(y = value, x = epoch, color = metric)) +
          geom_line() +
          labs(
            subtitle = paste0("KL for epochs ",
                              report_epochs[start_plot_row_ind],
                              " through ",
                              report_epochs[row_ind])
          )
        grid_plt <- gridExtra::grid.arrange(
          grobs = list(mse_plt, kl_plt),
          nrow = 2
        )
        print(grid_plt)
      } # end graphical training updates (want_plots = TRUE)
    } # end training updates (verbose = TRUE)
    
    # increment epoch counter
    epoch <- epoch + 1
    stop_criteria_met <- epoch > train_epochs
  } # end training WHILE loop
  
  ## compile results ----
  sim_res <- list(
    "sim_ind" = sim_ind,
    "loss_mat" = loss_mat,
    "alpha_mat" = alpha_mat
  )
  
  if (want_all_params){
    # add optionally stored network params
    sim_res$w_mu_mat <- w_mu_mat
    sim_res$w_var_mat <- w_var_mat
    sim_res$global_dropout_vec <- global_dropout_vec
    sim_res$marginal_dropout_mat <- marginal_dropout_mat
    sim_res$atilde_mu_mat <- atilde_mu_mat
    sim_res$btilde_mu_mat <- btilde_mu_mat
    sim_res$atilde_logvar_mat <- atilde_logvar_mat
    sim_res$btilde_logvar_mat <- btilde_logvar_mat
    sim_res$sa_mu_vec <- sa_mu_vec
    sim_res$sb_mu_vec <- sb_mu_vec
    sim_res$sa_logvar_vec <- sa_logvar_vec
    sim_res$sb_logvar_vec <- sb_logvar_vec
    # sim_res$tau_vec <- tau_vec
  } 
  
  if (want_data){
    # add data
    sim_res$x = as_array(lin_simdat$x)
    sim_res$y = as_array(lin_simdat$y)
    sim_res$true_coefs = lin_simdat$true_coefs
  }
  
  completed_msg <- paste0("\n \n ******************** \n sim #", 
                          sim_ind, 
                          " completed \n ",
                          "final alphas: ",
                          paste0(round(as_array(dropout_alphas), 3), collapse = ", "),
                          "\n ********************\n \n")
  cat_color(txt = completed_msg)
  return(sim_res)
}


# fcn1 <- function(x) exp(x/2)
# fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
# fcn3 <- function(x) abs(x)^(1.5)
# fcn4 <- function(x) - (abs(x))
# flist = list(fcn1, fcn2, fcn3, fcn4)

sim_fcn_hshoe_fcnaldata <- function(
    sim_ind,    # to ease parallelization
    sim_params,     # same as before, but need to include flist
    nn_model,   # torch nn_module,
    train_epochs = 10000,
    verbose = TRUE,      # provide updates in console
    display_alpha_thresh = 0.1,    # display alphas below this number
    report_every = 1000, # training epochs between display/store results
    want_plots = TRUE,   # provide graphical updates of KL, MSE
    want_fcn_plots = TRUE, # display predicted functions
    save_fcn_plots = FALSE,
    want_all_params = FALSE,
    save_mod = TRUE,
    save_mod_path_stem = NULL,
    stop_k = 100,
    stop_streak = 25,
    burn_in = 5E5
){
  # - fcn used with lapply for parallel processing
  # - basic linear regression setting
  # - only stop criteria used is max epochs
  # - if want_all_params = FALSE, only returns
  #     loss_mat (KL, mse vs epoch) and
  #     alpha_mat (local dropout rates vs epoch)
  # USAGE EXAMPLE:
  # > sim_params <- list(
  # >   "sim_name" = "horseshoe, linear regression setting, KL scaled by n",
  # >   "n_sims" = 100,
  # >   "d_in" = 104,
  # >   "n_obs" = 125,
  # >   "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  # >   "seed" = 314,
  # >   "err_sig" = 1,
  # >   "ttsplit" = 4/5,
  # >   "stop_criteria" = c("test_train","ma_loss_increasing")
  # > )
  # >
  # > res <- lapply(
  # >   1:sim_params$n_sims, 
  # >   function(X) sim_fcn_hshoe_linreg(
  # >     sim_ind = X, 
  # >     sim_params = sim_params,
  # >     nn_model = SLHS,  #### this needs to be pre-defined via torch::nn_module()
  # >     train_epochs = 1000,
  # >     verbose = TRUE,
  # >     report_every = 100,
  # >     want_plots = TRUE,
  # >     want_all_params = FALSE,
  # >     want_data = FALSE
  # >   )
  # > )
  
  ## generate data ----
  set.seed(sim_params$sim_seeds[sim_ind])
  torch_manual_seed(sim_params$sim_seeds[sim_ind])
  
  simdat <- sim_func_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    err_sigma = sim_params$err_sig
  )
  if (sim_params$use_cuda){
    simdat$x <- simdat$x$to(device = "cuda")
    simdat$y <- simdat$y$to(device = "cuda")
  }
  
  ## initialize BNN & optimizer ----
  model_fit <- nn_model()
  optim_model_fit <- optim_adam(model_fit$parameters)
  
  if (save_mod){
    if(is.null(save_mod_path_stem)){
      save_mod_path_stem <- here::here("sims", 
                                  "results", 
                                  paste0("fcnl_hshoe_mod_", 
                                         sim_params$n_obs, "obs_", 
                                         sim_params$sim_seeds[sim_ind]
                                         ))
    }
    save_mod_path <- paste0(save_mod_path_stem, ".pt")
  }
  
  
  
  ## set up plotting while training: ----
  # original function plots
  xshow <- seq(-3, 3, length.out = 100)
  yshow <- sapply(sim_params$flist, function(fcn) fcn(xshow))
  colnames(yshow) <- paste0("f", 1:ncol(yshow))
  
  orig_func_df <- data.frame(
    yshow,
    "x"  = xshow
  ) %>% 
    pivot_longer(cols = -x, names_to = "fcn")
  
  # to feed into BNN model during training loop updates
  curvmat <- matrix(0, ncol = length(sim_params$flist), nrow = length(sim_params$flist) * 100)
  for (i in 1:length(flist)){
    curvmat[1:length(xshow) + (i-1) * length(xshow), i] <- xshow
  }
  mat0 <- matrix(0, nrow = nrow(curvmat), ncol = sim_params$d_in - length(sim_params$flist))
  x_plot <- torch_tensor(cbind(curvmat, mat0), device = ifelse(sim_params$use_cuda, "cuda", "cpu"))
  y_plot <- model_fit(x_plot)  # need to add deterministic argument
  plotmat <- cbind(scale(as_array(y_plot)), curvmat)
  colnames(plotmat) <- c("y", paste0("f", 1:length(sim_params$flist)))
  plotdf <- as.data.frame(plotmat)
  
  
  # store: # train, test mse and kl ----
  report_epochs <- seq(
    report_every, 
    train_epochs, 
    by = report_every
  )
  
  loss_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = 3
  )
  colnames(loss_mat) <- c("kl", "mse_train", "mse_test")
  rownames(loss_mat) <- report_epochs
  
  
  # store: alphas
  alpha_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = sim_params$d_in
  )
  rownames(alpha_mat) <- report_epochs
  
  # store: weight posterior params
  if (want_all_params){
    w_mu_mat <-
      w_var_mat <- 
      marginal_dropout_mat <-
      atilde_mu_mat <- 
      btilde_mu_mat <-
      atilde_logvar_mat <-
      btilde_logvar_mat <- alpha_mat
    
    global_dropout_vec <-   
      # tau_vec <- 
      sa_mu_vec <-
      sb_mu_vec <- 
      sa_logvar_vec <- 
      sb_logvar_vec <- rep(NA, length(report_epochs))
  }
  
  
  ## SETUP test-train split ----
  ttsplit_ind <- floor(sim_params$n_obs * sim_params$ttsplit)
  ttsplit_used <- sim_params$ttsplit != 1
  x_train <- simdat$x[1:ttsplit_ind, ]
  y_train <- simdat$y[1:ttsplit_ind, ]
  if (ttsplit_used){
    x_test <- simdat$x[(ttsplit_ind+1):sim_params$n_obs, ] 
    y_test <- simdat$y[(ttsplit_ind+1):sim_params$n_obs, ]
  }
  
  
  ## TRAINING LOOP ----
  ## initialize training params
  epoch <- 1
  loss <- torch_tensor(1, device = dev_select(sim_params$use_cuda))
  
  ## stop criteria
  stop_criteria_met <- FALSE
  stop_epochs <- c()
  ## test_mse_storage
  mse_test <- torch_tensor(0, device = dev_select(sim_params$use_cuda))
  loss_test <- torch_tensor(1, device = dev_select(sim_params$use_cuda)) 
  test_mse_store <- rep(0, times = stop_k + 1)
  test_mse_streak <- rep(FALSE, times = stop_streak)
  
  
  while (!stop_criteria_met){
    prev_loss <- loss
    
    # fit & metrics
    yhat_train <- model_fit(x_train)
    mse <- nnf_mse_loss(yhat_train, y_train)
    kl <- model_fit$get_model_kld() / ttsplit_ind
    loss <- mse + kl
    
    
    # gradient step 
    # zero out previous gradients
    optim_model_fit$zero_grad()
    # backprop
    loss$backward()
    # update weights
    optim_model_fit$step()
    
    
    # compute test loss 
    if (ttsplit_used) {
      prev_loss_test <- loss_test
      yhat_test <- model_fit(x_test)
      mse_test <- nnf_mse_loss(yhat_test, y_test)
      if (epoch > (burn_in - 2 * (stop_k + stop_streak))) {
        test_mse_store <- roll_vec(test_mse_store, as_array(mse_test))
      }
    }
    
    
    # check test_mse stop criteria:
    # abs(diff(mse)) < sd(mse) for streak epochs in a row,
    # sd calculated using last stop_k epochs
    if (epoch > burn_in & ttsplit_used){
      test_mse_sd <- sd(test_mse_store[1:stop_k])
      test_mse_absdiff <- abs(diff(test_mse_store[stop_k + 0:1]))
      test_mse_compare <- test_mse_absdiff < test_mse_sd
      test_mse_streak <- roll_vec(test_mse_streak, test_mse_compare)
      test_mse_stopcrit <- all(test_mse_streak)
      
      if (test_mse_stopcrit){
        stop_epochs <- c(stop_epochs, epoch)
      }
    }
    
    
    ### store results (every `report_every` epochs) ----
    time_to_report <- epoch!=0 & (epoch %% report_every == 0)
    if (time_to_report){
      row_ind <- epoch %/% report_every
      
      loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_test$item())
      dropout_alphas <- model_fit$fc1$get_dropout_rates()
      alpha_mat[row_ind, ] <- as_array(dropout_alphas)
      
      # storing other optional parameters, mostly for diagnostics
      if (want_all_params){
        global_dropout_vec[row_ind] <- as_array(model_fit$fc1$get_dropout_rates(type = "global"))
        marginal_dropout_mat[row_ind, ] <- as_array(model_fit$fc1$get_dropout_rates(type = "marginal"))
        # tau_vec[row_ind] <- as_array(model_fit$fc1$tau)
        w_mu_mat[row_ind, ] <- as_array(model_fit$fc1$compute_posterior_param()$post_weight_mu)
        w_var_mat[row_ind, ] <- as_array(model_fit$fc1$compute_posterior_param()$post_weight_var)
        
        atilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_mu)
        btilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_mu)
        atilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_logvar)
        btilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_logvar)
        sa_mu_vec[row_ind] <- as_array(model_fit$fc1$sa_mu)
        sa_logvar_vec[row_ind] <- as_array(model_fit$fc1$sa_logvar)
        sb_mu_vec[row_ind] <- as_array(model_fit$fc1$sb_mu)
        sb_logvar_vec[row_ind] <- as_array(model_fit$fc1$sb_logvar)
      }
    } # end result storing and updating
    
    
    ### in-console and graphical training updates ----
    if (time_to_report & verbose){
      cat(
        "Epoch:", epoch,
        "MSE + KL/n =", mse$item(), "+", kl$item(),
        "=", loss$item(),
        "\n",
        "train mse:", round(mse$item(), 4), 
        "; test_mse:", round(mse_test$item(), 4),
        "\n", 
        sep = " "
      )
      cat("alphas: ", round(as_array(dropout_alphas), 2), "\n")
      display_alphas <- ifelse(
        as_array(dropout_alphas) <= display_alpha_thresh,
        round(as_array(dropout_alphas), 3),
        "."
      )
      cat("alphas below ", round(display_alpha_thresh, 4), ": ")
      cat(display_alphas, sep = " ")
      
      if (length(stop_epochs > 0)){
        stop_msg <- paste0(
          "\n ********************************************* \n",
          "test_mse STOP CONDITION reached ", 
          length(stop_epochs), " times; ", 
          "min / max: ", min(stop_epochs), " / ", max(stop_epochs),
          "\n ********************************************* \n"
        )
        cat_color(stop_msg)
      }
      cat(" \n \n")
      
      
      # graphical training updates
      if (want_plots & row_ind > 5){
        
        # only show most recent (for scale of plot)
        start_plot_row_ind <- 1
        if (row_ind >= 20){
          start_plot_row_ind <- row_ind - 10
        } else if (row_ind >= 60){
          start_plot_row_ind <- row_ind - 30
        } else if (row_ind >= 100){
          start_plot_row_ind <- row_ind - 50
        }
        # set up plotting df
        temp_lmat <- cbind(
          loss_mat[start_plot_row_ind:row_ind,],
          "epoch" = report_epochs[start_plot_row_ind:row_ind]
        )
        
        mse_plotdf <- temp_lmat %>% 
          data.frame() %>% 
          pivot_longer(
            cols = -epoch,
            names_to = "metric"
          )
        
        # plot MSE vs epochs
        mse_plt <- mse_plotdf %>% 
          filter(metric != "kl") %>% 
          ggplot(aes(y = value, x = epoch, color = metric)) + 
          geom_line() + 
          labs(
            subtitle = paste0("MSE for epochs ", 
                              report_epochs[start_plot_row_ind],
                              " through ",
                              report_epochs[row_ind])
          )
        
        # plot KL vs epochs
        kl_plt <- mse_plotdf %>%
          filter(metric == "kl") %>%
          ggplot(aes(y = value, x = epoch, color = metric)) +
          geom_line() +
          labs(
            subtitle = paste0("KL for epochs ",
                              report_epochs[start_plot_row_ind],
                              " through ",
                              report_epochs[row_ind])
          )
        grid_plt <- gridExtra::grid.arrange(
          grobs = list(mse_plt, kl_plt),
          nrow = 2
        )
        print(grid_plt)
      } # end graphical training updates (want_plots = TRUE)
    } # end training updates (verbose = TRUE)
    
    # function plots
    if (time_to_report & want_fcn_plots){
      plotdf$y <- scale(as_array(model_fit(x_plot)))
      
      plt <- plotdf %>% 
        pivot_longer(cols = -y, values_to = "x", names_to = "fcn") %>%
        ggplot(aes(y = y, x = x, color = fcn)) + 
        geom_line() + 
        geom_line(
          data = orig_func_df,
          linetype = "dotted",
          aes(y = value, x = x, color = fcn)
        ) + 
        labs(
          title = "predicted (solid) and original (dotted) functions",
          subtitle = paste0("epoch: ", epoch)
        )
      
      if (save_fcn_plots){
        save_plt_path <- paste0(save_mod_path_stem, "_e", epoch, ".png")
        ggsave(filename = save_plt_path, plot = plt, height = 4, width = 6)
      } else {
        print(plt)
      }
    } # end function update plots (want_fcn_plots = TRUE)
    
    
    # increment epoch counter
    epoch <- epoch + 1
    stop_criteria_met <- epoch > train_epochs
  } # end training WHILE loop
  
  ## compile results ----
  sim_res <- list(
    "sim_ind" = sim_ind,
    "stop_epochs" = stop_epochs,
    "fcn_plt" = plt,
    "loss_mat" = loss_mat,
    "alpha_mat" = alpha_mat
  )
  
  if (want_all_params){
    # add optionally stored network params
    sim_res$w_mu_mat <- w_mu_mat
    sim_res$w_var_mat <- w_var_mat
    sim_res$global_dropout_vec <- global_dropout_vec
    sim_res$marginal_dropout_mat <- marginal_dropout_mat
    sim_res$atilde_mu_mat <- atilde_mu_mat
    sim_res$btilde_mu_mat <- btilde_mu_mat
    sim_res$atilde_logvar_mat <- atilde_logvar_mat
    sim_res$btilde_logvar_mat <- btilde_logvar_mat
    sim_res$sa_mu_vec <- sa_mu_vec
    sim_res$sb_mu_vec <- sb_mu_vec
    sim_res$sa_logvar_vec <- sa_logvar_vec
    sim_res$sb_logvar_vec <- sb_logvar_vec
    # sim_res$tau_vec <- tau_vec
  } 
  
  # notify completed training ----
  completed_msg <- paste0(
    "\n \n ******************** \n ******************** \n",
    "sim #", 
    sim_ind, 
    " completed \n ",
    "final alphas below threshold: ",
    paste0(display_alphas, collapse = " "),
    "\n ******************** \n ******************** \n \n"
  )
  cat_color(txt = completed_msg)
  
  # save torch model ----
  if (save_mod){
    torch_save(model_fit, path = save_mod_path)
    cat_color(txt = paste0("model saved: ", save_mod_path))
    sim_res$mod_path = save_mod_path
  }
  return(sim_res)
}

sim_continue_training <- function(
    sim_seed,
    sim_params,     # same as before, but need to include flist
    nn_model,   # torch nn_module,
    verbose = TRUE,   # provide updates in console
    want_plots = TRUE,   # provide graphical updates of KL, MSE
    want_fcn_plots = TRUE,   # display predicted functions
    save_fcn_plots = FALSE,
    want_all_params = FALSE,
    save_mod = TRUE,
    save_results = TRUE,
    save_mod_path_stem = NULL
){
  ## generate data ----
  set.seed(sim_seed)
  torch_manual_seed(sim_seed)
  
  simdat <- sim_func_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    err_sigma = sim_params$err_sig
  )
  if (sim_params$use_cuda){
    simdat$x <- simdat$x$to(device = "cuda")
    simdat$y <- simdat$y$to(device = "cuda")
  }
  
  ## initialize BNN & optimizer ----
  model_fit <- nn_model
  optim_model_fit <- optim_adam(model_fit$parameters)
  
  
  if(is.null(save_mod_path_stem)){
    save_mod_path_stem <- here::here("sims", 
                                     "results", 
                                     paste0("fcnl_hshoe_mod_", 
                                            sim_params$n_obs, "obs_", 
                                            sim_seed, 
                                            "_continue"
                                     ))
  }
  save_mod_path <- paste0(save_mod_path_stem, ".pt")
  
  
  
  
  ## set up plotting while training: ----
  # original function plots
  xshow <- seq(-3, 3, length.out = 100)
  yshow <- sapply(sim_params$flist, function(fcn) fcn(xshow))
  colnames(yshow) <- paste0("f", 1:ncol(yshow))
  
  orig_func_df <- data.frame(
    yshow,
    "x"  = xshow
  ) %>% 
    pivot_longer(cols = -x, names_to = "fcn")
  
  # to feed into BNN model during training loop updates
  curvmat <- matrix(0, ncol = length(sim_params$flist), nrow = length(sim_params$flist) * 100)
  for (i in 1:length(flist)){
    curvmat[1:length(xshow) + (i-1) * length(xshow), i] <- xshow
  }
  mat0 <- matrix(0, nrow = nrow(curvmat), ncol = sim_params$d_in - length(sim_params$flist))
  x_plot <- torch_tensor(cbind(curvmat, mat0), device = ifelse(sim_params$use_cuda, "cuda", "cpu"))
  y_plot <- model_fit(x_plot)  # need to add deterministic argument
  plotmat <- cbind(scale(as_array(y_plot)), curvmat)
  colnames(plotmat) <- c("y", paste0("f", 1:length(sim_params$flist)))
  plotdf <- as.data.frame(plotmat)
  
  
  # store: # train, test mse and kl ----
  report_epochs <- seq(
    sim_params$report_every, 
    sim_params$train_epochs, 
    by = sim_params$report_every
  )
  
  loss_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = 3
  )
  colnames(loss_mat) <- c("kl", "mse_train", "mse_test")
  rownames(loss_mat) <- report_epochs
  
  
  # store: alphas
  alpha_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = sim_params$d_in
  )
  rownames(alpha_mat) <- report_epochs
  
  # store: weight posterior params
  if (want_all_params){
    w_mu_mat <-
      w_var_mat <- 
      marginal_dropout_mat <-
      atilde_mu_mat <- 
      btilde_mu_mat <-
      atilde_logvar_mat <-
      btilde_logvar_mat <- alpha_mat
    
    global_dropout_vec <-   
      # tau_vec <- 
      sa_mu_vec <-
      sb_mu_vec <- 
      sa_logvar_vec <- 
      sb_logvar_vec <- rep(NA, length(report_epochs))
  }
  
  
  ## SETUP test-train split and minibatching----
  ttsplit_ind <- floor(sim_params$n_obs * sim_params$ttsplit)
  x_train <- simdat$x[1:ttsplit_ind, ] 
  y_train <- simdat$y[1:ttsplit_ind, ]
  x_test <- simdat$x[(ttsplit_ind+1):sim_params$n_obs, ] 
  y_test <- simdat$y[(ttsplit_ind+1):sim_params$n_obs, ]
  
  if (!is.null(sim_params$batch_size)){
    num_batches <- ttsplit_ind %/% sim_params$batch_size
    batch_inds_vec <- 1:ttsplit_ind
    batch_size <- sim_params$batch_size
  } else {
    batch_size <- ttsplit_ind
  }
  
  
  ## TRAINING LOOP ----
  ## initialize training params
  epoch <- 1
  loss <- torch_tensor(1, device = dev_select(sim_params$use_cuda))
  
  ## stop criteria
  stop_criteria_met <- FALSE
  stop_epochs <- c()
  ## test_mse_storage
  mse_test <- torch_tensor(0, device = dev_select(sim_params$use_cuda))
  loss_test <- torch_tensor(1, device = dev_select(sim_params$use_cuda)) 
  test_mse_store <- rep(0, times = sim_params$stop_k + 1)
  test_mse_streak <- rep(FALSE, times = sim_params$stop_streak)
  
  
  while (!stop_criteria_met){
    
    # fit model with / without batching
    if (is.null(sim_params$batch_size)){
      yhat_train <- model_fit(x_train)
      mse <- nnf_mse_loss(yhat_train, y_train)
    } else {
      batch_num <- epoch %% num_batches
      # reshuffle batches if batch_num == 0
      if (batch_num == 0) {batch_inds_vec <- sample(1:ttsplit_ind)}
      batch_inds <- batch_inds_vec[1:sim_params$batch_size + (batch_num)*sim_params$batch_size]
      
      yhat_train <- model_fit(x_train[batch_inds, ])
      mse <- nnf_mse_loss(yhat_train, y_train[batch_inds])
    }
    
    
    # fit & metrics
    kl <- model_fit$get_model_kld() / batch_size
    loss <- mse + kl
    
    
    # gradient step 
    # zero out previous gradients
    optim_model_fit$zero_grad()
    # backprop
    loss$backward()
    # update weights
    optim_model_fit$step()
    
    
    # if (epoch > (sim_params$burn_in - 2 * (sim_params$stop_k + sim_params$stop_streak))) {
    #   test_mse_store <- roll_vec(test_mse_store, as_array(mse_test))
    # }
    # 
    #     # check test_mse stop criteria:
    #     # abs(diff(mse)) < sd(mse) for streak epochs in a row,
    #     # sd calculated using last stop_k epochs
    #     if (epoch > sim_params$burn_in){
    #       test_mse_sd <- sd(test_mse_store[1:sim_params$stop_k])
    #       test_mse_absdiff <- abs(diff(test_mse_store[sim_params$stop_k + 0:1]))
    #       test_mse_compare <- test_mse_absdiff < test_mse_sd
    #       test_mse_streak <- roll_vec(test_mse_streak, test_mse_compare)
    #       test_mse_stopcrit <- all(test_mse_streak)
    #       
    #       if (test_mse_stopcrit){
    #         stop_epochs <- c(stop_epochs, epoch)
    #       }
    #     }
    
    # store results (every `report_every` epochs)
    time_to_report <- epoch!=0 & (epoch %% sim_params$report_every == 0)
    if (time_to_report){
      row_ind <- epoch %/% sim_params$report_every
      
      # compute test loss 
      yhat_test <- model_fit(x_test)
      mse_test <- nnf_mse_loss(yhat_test, y_test)
      
      loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_test$item())
      dropout_alphas <- model_fit$fc1$get_dropout_rates()
      alpha_mat[row_ind, ] <- as_array(dropout_alphas)
      
      # storing other optional parameters, mostly for diagnostics
      if (want_all_params){
        global_dropout_vec[row_ind] <- as_array(model_fit$fc1$get_dropout_rates(type = "global"))
        marginal_dropout_mat[row_ind, ] <- as_array(model_fit$fc1$get_dropout_rates(type = "marginal"))
        # tau_vec[row_ind] <- as_array(model_fit$fc1$tau)
        w_mu_mat[row_ind, ] <- as_array(model_fit$fc1$compute_posterior_param()$post_weight_mu)
        w_var_mat[row_ind, ] <- as_array(model_fit$fc1$compute_posterior_param()$post_weight_var)
        
        atilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_mu)
        btilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_mu)
        atilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_logvar)
        btilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_logvar)
        sa_mu_vec[row_ind] <- as_array(model_fit$fc1$sa_mu)
        sa_logvar_vec[row_ind] <- as_array(model_fit$fc1$sa_logvar)
        sb_mu_vec[row_ind] <- as_array(model_fit$fc1$sb_mu)
        sb_logvar_vec[row_ind] <- as_array(model_fit$fc1$sb_logvar)
      }
    } # end result storing and updating
    
    
    ### in-console and graphical training updates ----
    if (!time_to_report & verbose & (epoch %% sim_params$report_every == 1)){cat("Training:")}
    if (!time_to_report & verbose & (epoch %% 100 == 0)){cat("#")}
    if (time_to_report & verbose){
      cat(
        "Epoch:", epoch,
        "MSE + KL/n =", mse$item(), "+", kl$item(),
        "=", loss$item(),
        "\n",
        "train mse:", round(mse$item(), 4), 
        "; test_mse:", round(mse_test$item(), 4),
        "\n", 
        sep = " "
      )
      cat("alphas: ", round(as_array(dropout_alphas), 2), "\n")
      display_alphas <- ifelse(
        as_array(dropout_alphas) <= sim_params$alpha_thresh,
        round(as_array(dropout_alphas), 3),
        "."
      )
      cat("alphas below ", round(sim_params$alpha_thresh, 4), ": ")
      cat(display_alphas, sep = " ")
      
      # if (length(stop_epochs > 0)){
      #   stop_msg <- paste0(
      #     "\n ********************************************* \n",
      #     "test_mse STOP CONDITION reached ", 
      #     length(stop_epochs), " times; ", 
      #     "min / max: ", min(stop_epochs), " / ", max(stop_epochs),
      #     "\n ********************************************* \n"
      #   )
      #   cat_color(stop_msg)
      # }
      cat(" \n \n")
      
      
      # graphical training updates
      if (want_plots & row_ind > 5){
        
        # only show most recent (for scale of plot)
        start_plot_row_ind <- 1
        if (row_ind >= 20){
          start_plot_row_ind <- row_ind - 10
        } else if (row_ind >= 60){
          start_plot_row_ind <- row_ind - 30
        } else if (row_ind >= 100){
          start_plot_row_ind <- row_ind - 50
        }
        # set up plotting df
        temp_lmat <- cbind(
          loss_mat[start_plot_row_ind:row_ind,],
          "epoch" = report_epochs[start_plot_row_ind:row_ind]
        )
        
        mse_plotdf <- temp_lmat %>% 
          data.frame() %>% 
          pivot_longer(
            cols = -epoch,
            names_to = "metric"
          )
        
        # plot MSE vs epochs
        mse_plt <- mse_plotdf %>% 
          filter(metric != "kl") %>% 
          ggplot(aes(y = value, x = epoch, color = metric)) + 
          geom_line() + 
          labs(
            subtitle = paste0("MSE for epochs ", 
                              report_epochs[start_plot_row_ind],
                              " through ",
                              report_epochs[row_ind])
          )
        
        # plot KL vs epochs
        kl_plt <- mse_plotdf %>%
          filter(metric == "kl") %>%
          ggplot(aes(y = value, x = epoch, color = metric)) +
          geom_line() +
          labs(
            subtitle = paste0("KL for epochs ",
                              report_epochs[start_plot_row_ind],
                              " through ",
                              report_epochs[row_ind])
          )
        grid_plt <- gridExtra::grid.arrange(
          grobs = list(mse_plt, kl_plt),
          nrow = 2
        )
        print(grid_plt)
      } # end graphical training updates (want_plots = TRUE)
    } # end training updates (verbose = TRUE)
    
    # function plots
    if (time_to_report & want_fcn_plots){
      plotdf$y <- scale(as_array(model_fit(x_plot)))
      
      plt <- plotdf %>% 
        pivot_longer(cols = -y, values_to = "x", names_to = "fcn") %>%
        ggplot(aes(y = y, x = x, color = fcn)) + 
        geom_line() + 
        geom_line(
          data = orig_func_df,
          linetype = "dotted",
          aes(y = value, x = x, color = fcn)
        ) + 
        labs(
          title = "predicted (solid) and original (dotted) functions",
          subtitle = paste0("epoch: ", epoch)
        )
      
      if (save_fcn_plots){
        save_plt_path <- paste0(save_mod_path_stem, "_e", epoch, ".png")
        ggsave(filename = save_plt_path, plot = plt, height = 4, width = 6)
      } else {
        print(plt)
      }
    } # end function update plots (want_fcn_plots = TRUE)
    
    
    # increment epoch counter
    epoch <- epoch + 1
    stop_criteria_met <- epoch > sim_params$train_epochs
  } # end training WHILE loop
  
  ## compile results ----
  sim_res <- list(
    "sim_seed" = sim_seed,
    # "stop_epochs" = stop_epochs,
    "fcn_plt" = plt,
    "loss_mat" = loss_mat,
    "alpha_mat" = alpha_mat
  )
  
  if (want_all_params){
    # add optionally stored network params
    sim_res$w_mu_mat <- w_mu_mat
    sim_res$w_var_mat <- w_var_mat
    sim_res$global_dropout_vec <- global_dropout_vec
    sim_res$marginal_dropout_mat <- marginal_dropout_mat
    sim_res$atilde_mu_mat <- atilde_mu_mat
    sim_res$btilde_mu_mat <- btilde_mu_mat
    sim_res$atilde_logvar_mat <- atilde_logvar_mat
    sim_res$btilde_logvar_mat <- btilde_logvar_mat
    sim_res$sa_mu_vec <- sa_mu_vec
    sim_res$sb_mu_vec <- sb_mu_vec
    sim_res$sa_logvar_vec <- sa_logvar_vec
    sim_res$sb_logvar_vec <- sb_logvar_vec
    # sim_res$tau_vec <- tau_vec
  } 
  
  ## notify completed training ----
  completed_msg <- paste0(
    "\n \n ******************** \n ******************** \n",
    "sim seed", 
    sim_seed, 
    " completed \n ",
    "final alphas below threshold: ",
    paste0(display_alphas, collapse = " "),
    "\n ******************** \n ******************** \n \n"
  )
  cat_color(txt = completed_msg)
  
  ## save torch model ----
  if (save_mod){
    torch_save(model_fit, path = save_mod_path)
    cat_color(txt = paste0("model saved: ", save_mod_path))
    sim_res$mod_path = save_mod_path
  }
  
  if (save_results){
    sim_res$sim_params <- sim_params
    save_res_path <- paste0(save_mod_path_stem, ".RData")
    save(sim_res, file = save_res_path)
    cat_color(txt = paste0("sim results saved: ", save_res_path))
  }
  return(sim_res)
}



sim_hshoe <- function(
    sim_ind = NULL,    # to ease parallelization
    seed = NULL,
    learning_rate = 0.001, # default for torch::optim_adam.  LUW uses 0.08 for NJ model
    sim_params,     # same as before, but need to include flist
    nn_model,   # torch nn_module,
    verbose = TRUE,   # provide updates in console
    want_plots = TRUE,   # provide graphical updates of KL, MSE
    want_fcn_plots = TRUE,   # display predicted functions
    save_fcn_plots = FALSE,
    want_all_params = FALSE,
    local_only = FALSE,
    save_mod = TRUE,
    save_results = TRUE,
    save_mod_path_stem = NULL
){
  if (is.null(seed)){
    seed <- sim_params$sim_seeds[sim_ind]
  }
  
  ## generate data ----
  set.seed(seed)
  torch_manual_seed(seed)
  
  simdat <- sim_func_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    err_sigma = sim_params$err_sig
  )
  if (sim_params$use_cuda){
    simdat$x <- simdat$x$to(device = "cuda")
    simdat$y <- simdat$y$to(device = "cuda")
  }
  
  ## initialize BNN & optimizer ----
  model_fit <- nn_model()
  optim_model_fit <- optim_adam(model_fit$parameters, lr = learning_rate)
  
  if (save_mod){
    if(is.null(save_mod_path_stem)){
      save_mod_path_stem <- here::here("sims", 
                                       "results", 
                                       paste0("fcnl_hshoe_mod_", 
                                              sim_params$n_obs, "obs_", 
                                              seed
                                       ))
    }
    save_mod_path <- paste0(save_mod_path_stem, ".pt")
  }
  
  
  
  ## set up plotting while training: ----
  # original function plots
  xshow <- seq(-3, 3, length.out = 100)
  yshow <- sapply(sim_params$flist, function(fcn) fcn(xshow))
  colnames(yshow) <- paste0("f", 1:ncol(yshow))
  
  orig_func_df <- data.frame(
    yshow,
    "x"  = xshow
  ) %>% 
    pivot_longer(cols = -x, names_to = "fcn")
  
  # to feed into BNN model during training loop updates
  curvmat <- matrix(0, ncol = length(sim_params$flist), nrow = length(sim_params$flist) * 100)
  for (i in 1:length(sim_params$flist)){
    curvmat[1:length(xshow) + (i-1) * length(xshow), i] <- xshow
  }
  mat0 <- matrix(0, nrow = nrow(curvmat), ncol = sim_params$d_in - length(sim_params$flist))
  x_plot <- torch_tensor(cbind(curvmat, mat0), device = ifelse(sim_params$use_cuda, "cuda", "cpu"))
  y_plot <- model_fit(x_plot)  # need to add deterministic argument
  plotmat <- cbind(scale(as_array(y_plot)), curvmat)
  colnames(plotmat) <- c("y", paste0("f", 1:length(sim_params$flist)))
  plotdf <- as.data.frame(plotmat)
  
  
  # store: # train, test mse and kl ----
  report_epochs <- seq(
    sim_params$report_every, 
    sim_params$train_epochs, 
    by = sim_params$report_every
  )
  
  loss_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = 3
  )
  colnames(loss_mat) <- c("kl", "mse_train", "mse_test")
  rownames(loss_mat) <- report_epochs
  
  
  # store: alphas, kappas
  alpha_mat <- matrix(
    NA, 
    nrow = length(report_epochs),
    ncol = sim_params$d_in
  )
  rownames(alpha_mat) <- report_epochs
  kappa_local_mat <- kappa_mat <- alpha_mat
  
  # store: weight params
  if (want_all_params){
    w_mu_arr <-
      w_lvar_arr <- array(NA, dim = c(sim_params$d_hidden1, sim_params$d_in, length(report_epochs)))
    
    atilde_mu_mat <- 
      btilde_mu_mat <-
      atilde_logvar_mat <-
      btilde_logvar_mat <- alpha_mat

    if (!local_only){
      sa_mu_vec <-
      sb_mu_vec <- 
      sa_logvar_vec <- 
      sb_logvar_vec <- rep(NA, length(report_epochs))
    }
  }
  
  
  ## SETUP test-train split and minibatching----
  ttsplit_ind <- floor(sim_params$n_obs * sim_params$ttsplit)
  x_train <- simdat$x[1:ttsplit_ind, ] 
  y_train <- simdat$y[1:ttsplit_ind, ]
  x_test <- simdat$x[(ttsplit_ind+1):sim_params$n_obs, ] 
  y_test <- simdat$y[(ttsplit_ind+1):sim_params$n_obs, ]
  
  if (!is.null(sim_params$batch_size)){
    num_batches <- ttsplit_ind %/% sim_params$batch_size
    batch_inds_vec <- 1:ttsplit_ind
    batch_size <- sim_params$batch_size
  } else {
    batch_size <- ttsplit_ind
  }
  
  
  ## TRAINING LOOP ----
  ## initialize training params
  epoch <- 1
  loss <- torch_tensor(1, device = dev_select(sim_params$use_cuda))
  
  ## stop criteria
  stop_criteria_met <- FALSE
  stop_epochs <- c()
  ## test_mse_storage
  mse_test <- torch_tensor(0, device = dev_select(sim_params$use_cuda))
  loss_test <- torch_tensor(1, device = dev_select(sim_params$use_cuda)) 
  test_mse_store <- rep(0, times = sim_params$stop_k + 1)
  test_mse_streak <- rep(FALSE, times = sim_params$stop_streak)
  
  
  while (!stop_criteria_met){
    
    # fit model with / without batching
    if (is.null(sim_params$batch_size)){
      yhat_train <- model_fit(x_train)
      mse <- nnf_mse_loss(yhat_train, y_train)
    } else {
      batch_num <- epoch %% num_batches
      # reshuffle batches if batch_num == 0
      if (batch_num == 0) {batch_inds_vec <- sample(1:ttsplit_ind)}
      batch_inds <- batch_inds_vec[1:sim_params$batch_size + (batch_num)*sim_params$batch_size]
      
      yhat_train <- model_fit(x_train[batch_inds, ])
      mse <- nnf_mse_loss(yhat_train, y_train[batch_inds])
    }

    
    # fit & metrics
    kl <- model_fit$get_model_kld() / batch_size
    loss <- mse + kl
    
    
    # gradient step 
    # zero out previous gradients
    optim_model_fit$zero_grad()
    # backprop
    loss$backward()
    # update weights
    optim_model_fit$step()
    
    
    # if (epoch > (sim_params$burn_in - 2 * (sim_params$stop_k + sim_params$stop_streak))) {
    #   test_mse_store <- roll_vec(test_mse_store, as_array(mse_test))
    # }
    # 
    #     # check test_mse stop criteria:
    #     # abs(diff(mse)) < sd(mse) for streak epochs in a row,
    #     # sd calculated using last stop_k epochs
    #     if (epoch > sim_params$burn_in){
    #       test_mse_sd <- sd(test_mse_store[1:sim_params$stop_k])
    #       test_mse_absdiff <- abs(diff(test_mse_store[sim_params$stop_k + 0:1]))
    #       test_mse_compare <- test_mse_absdiff < test_mse_sd
    #       test_mse_streak <- roll_vec(test_mse_streak, test_mse_compare)
    #       test_mse_stopcrit <- all(test_mse_streak)
    #       
    #       if (test_mse_stopcrit){
    #         stop_epochs <- c(stop_epochs, epoch)
    #       }
    #     }
    
    # store results (every `report_every` epochs)
    time_to_report <- epoch!=0 & (epoch %% sim_params$report_every == 0)
    if (time_to_report){
      row_ind <- epoch %/% sim_params$report_every
      
      # compute test loss 
      yhat_test <- model_fit(x_test)
      mse_test <- nnf_mse_loss(yhat_test, y_test)
      
      loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_test$item())
      dropout_alphas <- model_fit$fc1$get_dropout_rates()
      alpha_mat[row_ind, ] <- as_array(dropout_alphas)
      kappas <- get_kappas(model_fit$fc1)
      kappa_mat[row_ind, ] <- kappas
      
      kappas_local <- get_kappas(model_fit$fc1, type = "local")
      kappa_local_mat[row_ind, ] <- kappas_local
      
      # storing other optional parameters, mostly for diagnostics
      if (want_all_params){
        # global_dropout_vec[row_ind] <- as_array(model_fit$fc1$get_dropout_rates(type = "global"))
        # marginal_dropout_mat[row_ind, ] <- as_array(model_fit$fc1$get_dropout_rates(type = "marginal"))
        # # tau_vec[row_ind] <- as_array(model_fit$fc1$tau)
        w_mu_arr[, , row_ind] <- as_array(model_fit$fc1$weight_mu)
        w_lvar_arr[, , row_ind] <- as_array(model_fit$fc1$weight_logvar)
        
        atilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_mu)
        btilde_mu_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_mu)
        atilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$atilde_logvar)
        btilde_logvar_mat[row_ind, ] <- as_array(model_fit$fc1$btilde_logvar)
        if (!local_only){
          sa_mu_vec[row_ind] <- as_array(model_fit$fc1$sa_mu)
          sa_logvar_vec[row_ind] <- as_array(model_fit$fc1$sa_logvar)
          sb_mu_vec[row_ind] <- as_array(model_fit$fc1$sb_mu)
          sb_logvar_vec[row_ind] <- as_array(model_fit$fc1$sb_logvar)
        }
      }
    } # end result storing and updating
    
    
    ### in-console and graphical training updates ----
    # visual updating on epochs
    # cat("\r", "Progress:", i, "%")
    if (!time_to_report & verbose & (epoch %% sim_params$report_every == 1)){cat("Training till next report:")}
    if (!time_to_report & verbose & (epoch %% round(sim_params$report_every/100) == 1)){cat("#")}
    
    # report
    if (time_to_report & verbose){
      cat(
        "\n Epoch:", epoch,
        "MSE + KL/n =", round(mse$item(), 5), "+", round(kl$item(), 5),
        "=", round(loss$item(), 4),
        "\n",
        "train mse:", round(mse$item(), 4), 
        "; test_mse:", round(mse_test$item(), 4),
        sep = " "
      )
      
      # report global shrinkage params (s^2 or tau^2)
      s_sq1 <- get_s_sq(model_fit$fc1)
      s_sq2 <- get_s_sq(model_fit$fc2)
      s_sq3 <- get_s_sq(model_fit$fc3)
      
      cat(
        "\n s_sq1 = ", round(s_sq1, 5),
        "; s_sq2 = ", round(s_sq2, 5),
        "; s_sq3 = ", round(s_sq3, 5),
        sep = ""
      )
      cat("\n alphas: ", round(as_array(dropout_alphas), 2), "\n")
      display_alphas <- ifelse(
        as_array(dropout_alphas) <= sim_params$alpha_thresh,
        round(as_array(dropout_alphas), 3),
        "."
      )
      cat("alphas below 0.82: ")
      cat(display_alphas, sep = " ")
      
      cat("\n global kappas: ", round(kappas, 2), "\n")
      display_kappas <- ifelse(
        kappas <= 0.9,
        round(kappas, 3),
        "."
      )
      cat("global kappas below 0.9: ")
      cat(display_kappas, sep = " ")
      
      cat("\n local kappas: ", round(kappas_local, 2), "\n")
      
      
      # if (length(stop_epochs > 0)){
      #   stop_msg <- paste0(
      #     "\n ********************************************* \n",
      #     "test_mse STOP CONDITION reached ", 
      #     length(stop_epochs), " times; ", 
      #     "min / max: ", min(stop_epochs), " / ", max(stop_epochs),
      #     "\n ********************************************* \n"
      #   )
      #   cat_color(stop_msg)
      # }
      cat(" \n \n")
      
      
      # graphical training updates
      if (want_plots & row_ind > 5){
        
        # only show most recent (for scale of plot)
        start_plot_row_ind <- 1
        if (row_ind >= 20){
          start_plot_row_ind <- row_ind - 10
        } else if (row_ind >= 60){
          start_plot_row_ind <- row_ind - 30
        } else if (row_ind >= 100){
          start_plot_row_ind <- row_ind - 50
        }
        # set up plotting df
        temp_lmat <- cbind(
          loss_mat[start_plot_row_ind:row_ind,],
          "epoch" = report_epochs[start_plot_row_ind:row_ind]
        )
        
        mse_plotdf <- temp_lmat %>% 
          data.frame() %>% 
          pivot_longer(
            cols = -epoch,
            names_to = "metric"
          )
        
        # plot MSE vs epochs
        mse_plt <- mse_plotdf %>% 
          filter(metric != "kl") %>% 
          ggplot(aes(y = value, x = epoch, color = metric)) + 
          geom_line() + 
          labs(
            subtitle = paste0("MSE for epochs ", 
                              report_epochs[start_plot_row_ind],
                              " through ",
                              report_epochs[row_ind])
          )
        
        # plot KL vs epochs
        kl_plt <- mse_plotdf %>%
          filter(metric == "kl") %>%
          ggplot(aes(y = value, x = epoch, color = metric)) +
          geom_line() +
          labs(
            subtitle = paste0("KL for epochs ",
                              report_epochs[start_plot_row_ind],
                              " through ",
                              report_epochs[row_ind])
          )
        grid_plt <- gridExtra::grid.arrange(
          grobs = list(mse_plt, kl_plt),
          nrow = 2
        )
        print(grid_plt)
      } # end graphical training updates (want_plots = TRUE)
    } # end training updates (verbose = TRUE)
    
    # function plots
    if (time_to_report & want_fcn_plots){
      plotdf$y <- scale(as_array(model_fit(x_plot)))
      
      plt <- plotdf %>% 
        pivot_longer(cols = -y, values_to = "x", names_to = "fcn") %>%
        ggplot(aes(y = y, x = x, color = fcn)) + 
        geom_line() + 
        geom_line(
          data = orig_func_df,
          linetype = "dotted",
          aes(y = value, x = x, color = fcn)
        ) + 
        labs(
          title = "predicted (solid) and original (dotted) functions",
          subtitle = paste0("epoch: ", epoch)
        )
      
      if (save_fcn_plots){
        save_plt_path <- paste0(save_mod_path_stem, "_e", epoch, ".png")
        ggsave(filename = save_plt_path, plot = plt, height = 4, width = 6)
      } else {
        print(plt)
      }
    } # end function update plots (want_fcn_plots = TRUE)
    
    
    # increment epoch counter
    epoch <- epoch + 1
    stop_criteria_met <- epoch > sim_params$train_epochs
  } # end training WHILE loop
  
  ### compile results ----
  sim_res <- list(
    "sim_ind" = sim_ind,
    # "stop_epochs" = stop_epochs,
    "fcn_plt" = plt,
    "loss_mat" = loss_mat,
    "alpha_mat" = alpha_mat,
    "kappa_mat" = kappa_mat,
    "kappa_local_mat" = kappa_local_mat
  )
  
  if (want_all_params){
    # add optionally stored network params
    sim_res$w_mu_arr <- w_mu_arr
    sim_res$w_lvar_arr <- w_lvar_arr
    # sim_res$global_dropout_vec <- global_dropout_vec
    # sim_res$marginal_dropout_mat <- marginal_dropout_mat
    sim_res$atilde_mu_mat <- atilde_mu_mat
    sim_res$btilde_mu_mat <- btilde_mu_mat
    sim_res$atilde_logvar_mat <- atilde_logvar_mat
    sim_res$btilde_logvar_mat <- btilde_logvar_mat
    if (!local_only){
      sim_res$sa_mu_vec <- sa_mu_vec
      sim_res$sb_mu_vec <- sb_mu_vec
      sim_res$sa_logvar_vec <- sa_logvar_vec
      sim_res$sb_logvar_vec <- sb_logvar_vec
    }
    # sim_res$tau_vec <- tau_vec
  } 
  
  ### notify completed training ----
  completed_msg <- paste0(
    "\n \n ******************** \n ******************** \n",
    "sim #", 
    sim_ind, 
    " completed \n ",
    "final alphas below threshold: ",
    paste0(display_alphas, collapse = " "),
    "\n ******************** \n ******************** \n \n"
  )
  cat_color(txt = completed_msg)
  
  ### save torch model & sim results ----
  if (save_mod){
    torch_save(model_fit, path = save_mod_path)
    cat_color(txt = paste0("model saved: ", save_mod_path))
    sim_res$mod_path = save_mod_path
  }
  if (save_results){
    sim_res$sim_params <- sim_params
    save_res_path <- paste0(save_mod_path_stem, ".RData")
    save(sim_res, file = save_res_path)
    cat_color(txt = paste0("sim results saved: ", save_res_path))
  }
  return(sim_res)
}





