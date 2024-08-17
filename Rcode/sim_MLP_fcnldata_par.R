##################################################
## Project:   MLP functional data sim
## Date:      Aug 14, 2024
## Author:    Arnie Seong
##################################################

library(here)
library(tidyr)
library(dplyr)
library(parallel)

# BayesCompress
library(torch)
source(here("Rcode", "BayesianLayers.R"))
source(here("Rcode", "sim_functions.R"))

#competitors
library(BoomSpikeSlab)
library(glmnet)
library(pbapply)



# parallelization params ----

on_server <- FALSE
ncpus <- detectCores()

cl_size <- ifelse(
  on_server,
  ncpus / 2,
  ncpus - 4
)

cl_size <- ifelse(cl_size < 1, 1, cl_size)


# sim params --------------------------------------------------------------
testing <- FALSE

fname <- ifelse(testing, "MLP_fcnl_sim_sig1_TEST", "MLP_fcnl_sim_sig1")
fpath <- here("Rcode", "results", paste0(fname, ".Rdata"))

n_sims <- ifelse(testing, 5, 100)
cl_size <- ifelse(testing, 2, cl_size)

n_sims_each_partial <- cl_size * ifelse(testing, 2, 5)
# separate save files for parallelization
num_saves <- n_sims %/% n_sims_each_partial + ifelse(n_sims %% n_sims_each_partial == 0, 0, 1)
partial_fpaths <- here("Rcode", "results", paste0(fname, "_PARTIAL", 1:num_saves, ".Rdata"))


# simulated data settings
n_obs <- 10000
sig <- 1
d_in <-  100


# generate data
fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x)
fcn3 <- function(x) abs(x)^(1.5)
fcn4 <- function(x) cos(pi*x) + sin(pi/1.2*x) - x

flist <- list(fcn1, fcn2, fcn3, fcn4)

fcn_simdat <- sim_func_data(
  n_obs = n_obs,
  d_in = d_in,
  flist = flist, 
  err_sigma = sig)

true_model <- c(
  rep(1, fcn_simdat$d_true), 
  rep(0, fcn_simdat$d_in-fcn_simdat$d_true)
)



# # # # # # # # # # # # 
##    MLNJ parameters
d_hidden1 <- 50
d_hidden2 <- 25
dropout_thresh <- 0.05

# set initial value for dropout rate alpha
# (default value is 1/2)
init_alpha <- 0.5
max_train_epochs <- ifelse(testing, 500, 40000)
verbose <- testing
burn_in <- ifelse(testing, 100, 10000)
convergence_crit <- 1e-6
# loss moving average stopping criterion length
ma_length <- 50





# # # # # # # # # # # # # # # # # # # # # # # # #
## SIM FUNCTION ----
# # # # # # # # # # # # # # # # # # # # # # # # #
MLP_fcnldata_sim <- function(
  n_obs,
  sig,
  flist,
  d_in,
  d_hidden1,
  d_hidden2,
  use_cuda,
  init_alpha,
  ma_length,
  max_train_epochs,
  verbose,
  burn_in,
  convergence_crit
){

  # # # # # # # # # # # # 
  ##    SIMULATION START    
  res <- list()
  
  fcn_simdat <- sim_func_data(
    n_obs = n_obs,
    d_in = d_in,
    flist = flist, 
    err_sigma = sig)

  true_model <- c(
    rep(1, fcn_simdat$d_true), 
    rep(0, fcn_simdat$d_in-fcn_simdat$d_true)
  )
  
  # # # # # # # # # # # # # # # # # # # # # # # # #
  ## BAYESIAN LAYER NJ ----
  # # # # # # # # # # # # # # # # # # # # # # # # #
  
  MLNJ <- nn_module(
    "MLNJ",
    initialize = function() {
      self$fc1 = BayesianLayerNJ(    
        in_features = fcn_simdat$d_in, 
        out_features = d_hidden1,
        use_cuda = use_cuda,
        init_weight = NULL,
        init_bias = NULL,
        init_alpha = init_alpha,
        clip_var = NULL
      )
      
      self$fc2 = BayesianLayerNJ(    
        in_features = d_hidden1, 
        out_features = d_hidden2,
        use_cuda = use_cuda,
        init_weight = NULL,
        init_bias = NULL,
        init_alpha = init_alpha,
        clip_var = NULL
      )
      
      self$fc3 = BayesianLayerNJ(    
        in_features =  d_hidden2, 
        out_features = 1,
        use_cuda = use_cuda,
        init_weight = NULL,
        init_bias = NULL,
        init_alpha = init_alpha,
        clip_var = NULL
      )
    },
    
    forward = function(x) {
      x %>% 
        self$fc1() %>%
        nnf_relu() %>%
        self$fc2() %>% 
        nnf_relu() %>% 
        self$fc3()
    },
    
    get_model_kld = function(){
      kl1 = self$fc1$get_kl()
      kl2 = self$fc2$get_kl()
      kl3 = self$fc3$get_kl()
      kld = kl1 + kl2 + kl3
      return(kld)
    }
  )
  
  mlnj_net <- MLNJ()
  optim_mlnj <- optim_adam(mlnj_net$parameters)
  
  
  # initialize stopping criteria params
  # 3 stopping criteria:
  #    1: epochs > max_train_epochs
  #    2: loss_diff < convergence crit
  #    3: loss moving average (average over last ma_length epochs) 
  #       has increased for the last ma_length epochs.
  #       In this case, best model occurred ma_length epochs ago.
  loss_diff <- 1
  loss <- torch_zeros(1)
  loss_ma_stop <- FALSE
  converge_stop <- FALSE
  
  # track loss, alphas for stopping criteria
  loss_store_mat <- array(NA, dim = c(3, 2*ma_length))
  rownames(loss_store_mat) <- c("epoch", "loss", "loss_MA")
  log_alpha_mat <- array(NA, dim = c(2*ma_length, d_in + 4))
  colnames(log_alpha_mat) <- c(
    paste0("x", 1:d_in),
    "epoch",
    "mse",
    "kl",
    "coef_mse"
  )
  
  epoch <- 0
  while (epoch < max_train_epochs & !converge_stop & !loss_ma_stop){
    prev_loss <- loss
    epoch <- epoch + 1
    
    y_pred <- mlnj_net(fcn_simdat$x)
    
    mse <- nnf_mse_loss(y_pred, fcn_simdat$y)
    kl <- mlnj_net$get_model_kld() / fcn_simdat$n_obs
    
    loss <- mse + kl
    loss_diff <- (loss - prev_loss)$item()
    
    if(epoch %% 1000 == 0 & verbose){
      cat(
        "Epoch: ", epoch, 
        "MSE + KL/n = ", mse$item(), " + ", kl$item(), 
        " = ", loss$item(), 
        "\n"
      )
      mlnj_keeps <- mlnj_net$fc1$get_log_dropout_rates()$exp() < 0.05
      mlnj_bin_err <- binary_err(est = as_array(mlnj_keeps), tru = true_model)
      cat("binary error: \n")
      cat(round(mlnj_bin_err, 4))
      cat("\n")
      
    }
    
    if (epoch > burn_in) {
      # stopping criterion --- if loss MA hasn't decreased
      # in ma_length epochs, stop training
      store_ind <- epoch %% (2*ma_length) + 1
      loss_store_mat[1, store_ind] <- epoch
      loss_store_mat[2, store_ind] <- loss$item()
      log_alpha_mat[store_ind, ] <- c(
        as_array(mlnj_net$fc1$get_log_dropout_rates()),
        epoch,
        mse$item(), 
        kl$item(),
        NA
      )
      
      # store loss moving average
      # subset losses to contribute to MA
      if (store_ind == 2*ma_length){
        ma_inds <- c(2*ma_length, 1:(ma_length-1))
      } else if (store_ind > ma_length){
        ma_inds <- c(
          store_ind:(2 * ma_length),
          1:((store_ind-ma_length) %% ma_length)
        )[1:ma_length]
      } else {
        ma_inds <- store_ind:(store_ind + ma_length - 1)
      }
      
      losses_vec <- loss_store_mat[2, ma_inds]
      loss_store_mat[3, store_ind] <- mean(losses_vec, na.rm = TRUE)
      
      if (epoch > (burn_in + 2*ma_length)){
        
        converge_stop <- abs(loss_diff) < convergence_crit
        
        loss_ma_vec <- loss_store_mat[3, ma_inds]
        loss_ma_diff <- diff(loss_ma_vec)
        if (sum(loss_ma_diff > 0) >= length(loss_ma_diff)) {
          loss_ma_stop <- TRUE
        }
      }
      
    }
    
    
    # zero out previous gradients
    optim_mlnj$zero_grad()
    # backprop
    loss$backward()
    # update weights
    optim_mlnj$step()
    
  }
  
  # best model's alphas / other metrics
  if (loss_ma_stop) {
    best_epoch_row <- which(log_alpha_mat[, d_in + 1] == epoch - (ma_length))
  } else {
    best_epoch_row <- which(log_alpha_mat[, d_in + 1] == epoch)
  }
  
  # in case which returns "integer(0)" above:
  if (length(best_epoch_row) == 0){
    best_epoch_row <- store_ind
  }
  
  log_dropout_alphas <- log_alpha_mat[best_epoch_row, 1:d_in]
  other_metrics <- c(log_alpha_mat[best_epoch_row, d_in + 1:4])
  stop_reason <- c(
    "max_epochs" = epoch > max_train_epochs,
    "converge_crit" = converge_stop,
    "loss_ma" = loss_ma_stop
  ) 
  
  
  
  if (verbose){
    mlnj_keeps <- exp(log_dropout_alphas) < dropout_thresh
    mlnj_bin_err <- binary_err(est = mlnj_keeps, tru = true_model)
    
    cat("binary error: \n")
    mlnj_bin_err
  }
  
  
  
  # store results: 
  # epoch
  # stop_reason
  # dropout_alphas
  # mse
  mlnj_res <- list(
    "epoch" = epoch,
    "stop_reason" = stop_reason,
    "log_dropout_alphas" = log_dropout_alphas,
    "log_alpha_mat" = log_alpha_mat,
    "other_metrics" = other_metrics
  )
  
  res$mlnj <- mlnj_res
  
  
  
  # # # # # # # # # # # # # # # # # # # # # # # # #
  ## lm ----
  # # # # # # # # # # # # # # # # # # # # # # # # #
  lm_res <- get_lm_stats(simdat = fcn_simdat)
  res$lm <- lm_res
  
  
  
  # # # # # # # # # # # # # # # # # # # # # # # # #
  ## LASSO ----
  # # # # # # # # # # # # # # # # # # # # # # # # #
  x <- as_array(fcn_simdat$x)
  y <- as_array(fcn_simdat$y)
  
  cvfit <- glmnet::cv.glmnet(x, y)
  lasso_coefs <- coef(cvfit, s = "lambda.1se")
  lasso_mse <- cvfit$cvm[cvfit$lambda == cvfit$lambda.1se]
  lasso_bin_err <- binary_err(est = !(lasso_coefs[-1]==0), tru = true_model)
  
  res$lasso <- list(
    "coefs" = lasso_coefs,
    "binary_err" = lasso_bin_err,
    "fit_mse" = lasso_mse,
    "coef_mse" = NA
  )
  
  
  
  
  
  # # # # # # # # # # # # # # # # # # # # # # # # #
  ## spike-slab ----
  # # # # # # # # # # # # # # # # # # # # # # # # #
  X <- cbind(1, x)
  prior = IndependentSpikeSlabPrior(X, y, 
                                    expected.model.size = 1,
                                    prior.beta.sd = rep(1, ncol(X))) 
  lm_ss = lm.spike(y ~ x, niter = 1000, prior = prior, ping = 0)
  ss_allcoefs_unordered <- summary(lm_ss)$coef
  
  # reorder results
  ss_intercept <- ss_allcoefs_unordered[rownames(ss_allcoefs_unordered)=="(Intercept)",]
  ss_coefs_unordered <- ss_allcoefs_unordered[rownames(ss_allcoefs_unordered)!="(Intercept)",]
  coef_order <- as.numeric(sapply(strsplit(rownames(ss_coefs_unordered), "x"), function(X) X[2]))
  ss_allcoefs <- rbind(
    "(Intercept)" = ss_intercept,
    ss_coefs_unordered[order(coef_order),]
  )
  ss_coefs <- ss_allcoefs[-1, ]
  ss_bin_err <- binary_err(
    est = ss_coefs[, 4] > 0.05, 
    tru = true_model)
  
  # get median mse from last quarter of draws
  ss_mse <- median(lm_ss$sse[751:1000]) / n_obs
  
  ss_res <- list(
    "coefs" = ss_coefs,
    "binary_err" = ss_bin_err,
    "fit_mse" = ss_mse,
    "coef_mse" = NA
  )
  
  res$ss <- ss_res
  
  return(res)
}






# # # # # # # # # # # # # # # # # # # # # # # # #
## RUN SIMULATION ----
# # # # # # # # # # # # # # # # # # # # # # # # #
# each partial save contains cl_size simulations

sim_params <- list(
  "n_obs" = n_obs,
  "sig" = sig,
  "d_in" = d_in,
  "d_hidden1" = d_hidden1,
  "d_hidden2" = d_hidden2,
  "init_alpha" = init_alpha,
  "ma_length" = ma_length,
  "max_train_epochs" = max_train_epochs,
  "verbose" = verbose,
  "burn_in" = burn_in,
  "convergence_crit" = convergence_crit,
  "n_sims" = n_sims,
  "true_model" = true_model,
  "MLP_fcnldata_sim" = MLP_fcnldata_sim
)

for (partial_num in 1:length(partial_fpaths)) {
  gc()
  # parallelize
  cl <- makeCluster(cl_size,
                    type="PSOCK",
                    outfile=paste0(fpath, "_monitor.txt"))
  
  # EXPORT variables, libraries
  # export libraries
  parallel::clusterEvalQ(cl = cl, library(here))
  parallel::clusterEvalQ(cl = cl, library(tidyr))
  parallel::clusterEvalQ(cl = cl, library(dplyr))
  parallel::clusterEvalQ(cl = cl, library(torch))
  parallel::clusterEvalQ(cl = cl, library(BoomSpikeSlab))
  parallel::clusterEvalQ(cl = cl, library(glmnet))

  
  # SOURCE functions
  parallel::clusterCall(cl, function() { source(here::here("Rcode", "BayesianLayers.R")) })
  parallel::clusterCall(cl, function() { source(here::here("Rcode", "sim_functions.R")) })

  # export variables
  parallel::clusterExport(
    cl = cl,
    envir = environment(),
    varlist = ls()
  )
  
  # Set seed
  parallel::clusterSetRNGStream(cl, iseed = 0L)
  
  # save partial results as diff files
  partial_res <- pblapply(
    1:n_sims_each_partial, 
    function(X)
      MLP_fcnldata_sim(
        n_obs,
        sig,
        flist,
        d_in,
        d_hidden1,
        d_hidden2,
        use_cuda = FALSE,
        init_alpha,
        ma_length,
        max_train_epochs,
        verbose,
        burn_in,
        convergence_crit
      ),
    cl = cl
  )
  save(partial_res, file = partial_fpaths[partial_num])
  txt <- paste0("saved partial_save ", partial_num, " of ", length(partial_fpaths))
  cat(txt)
  cat("\n")
  
  parallel::stopCluster(cl)
}
##### end parallelized simulations




# # # # # # # # # # # # 
##    stitch partial result files back together
result <- list()
for (partial_num in 1:length(partial_fpaths)){
  load(file = partial_fpaths[partial_num])
  result <- append(
    result,
    partial_res
  )
}


save(result, sim_params, file = fpath)
cat("\n simulation results saved \n")



# # check
# rm(result, sim_params)
# load(fpath)
# names(sim_params)
# 
# length(sim_params)
# sapply(result, function(X) X$mlnj$other_metrics)
# sapply(result, function(X) X$mlnj$log_dropout_alphas)









