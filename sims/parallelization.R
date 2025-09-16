
#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))


#### GPU acceleration? ----
if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}


#### data-generating functions ----
fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
fcn3 <- function(x) abs(x)^(1.5)
fcn4 <- function(x) - (abs(x))
flist = list(fcn1, fcn2, fcn3, fcn4)

# plot_datagen_fcns(flist)



#### sim_params ----
sim_params <- list(
  "sim_name" = "hot start, horseshoe, fcnal data",
  "seed" = 1002,
  "n_sims" = 2, 
  "train_epochs" = 75E4,
  "report_every" = 1E3,
  "use_cuda" = FALSE,    # use_cuda,
  "d_in" = 104,
  "d_hidden1" = 16,
  "d_hidden2" = 8,
  "d_out" = 1,
  "n_obs" = 12500,
  "true_coefs" = c(-0.5, 1, -2, 4, rep(0, times = 100)),
  "wald_thresh" = 1 / qchisq(1 - (0.05 / 104), df = 1),
  "flist" = flist,
  "err_sig" = 1,
  "convergence_crit" = 1e-7,
  "ttsplit" = 4/5,
  "stop_k" = 100,
  "stop_streak" = 25,
  "burn_in" = 2E5,
  "stop_criteria" = c(
    "test_train",        # [te_loss - tr_loss] positive & increasing for [stop_criteria_interval] epochs
    "train_convergence", # tr_loss_diff < [convergence_crit] for [stop_cruit_interval] epochs
    "test_convergence",  # te_loss_diff < [convergence_crit] for ...
    "ma_loss_increasing" # ma_tr_loss increasing for ...
  )
)

sim_params$hot_start_epochs <- 5E5

save_fname <- paste0(
  "hshoe_fcnl_multi",
  sim_params$n_obs,
  "_maxepochs",
  sim_params$seed,
  "_TESTING",
  ".RData"
)

set.seed(sim_params$seed)
sim_params$sim_seeds <- floor(runif(n = sim_params$n_sims, 0, 1000000))

# simulate data with set.seed and torch set seed ----
sim_ind <- 1   # comment out when not testing
set.seed(sim_params$sim_seeds[sim_ind])
torch_manual_seed(sim_params$sim_seeds[sim_ind])


hot_start_DNN <- function(
    sim_ind,
    sim_params,
    verbose = TRUE,
    want_fcn_plts = TRUE,
    save_mod = TRUE,
    save_path = here::here("sims", "results", "hotstart500k")
){
  # train regular DNN for preliminary weights to save training time
  # returns trained DNN; can also save trained DNN
  if (save_mod & is.null(save_path)){
    fname <- paste0("dnn_seed", sim_params$sim_seeds[sim_ind], ".pt")
    save_path <- here::here(fname)
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
  
  if (want_fcn_plts){
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
      if (want_fcn_plts){
        plt <- plot_fcn_preds(
          torchmod = DNN,
          pred_mats = pred_mats
        )
        print(plt)
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



DNN <- hot_start_DNN(
    sim_ind = 1,
    sim_params,
    save_mod = FALSE,
    save_path = NULL
)


# ## define model
# MLHS <- nn_module(
#   "MLHS",
#   initialize = function() {
#     self$fc1 = torch_hs(    
#       in_features = sim_params$d_in, 
#       out_features = sim_params$d_hidden1,
#       use_cuda = FALSE,
#       tau = 1,
#       init_weight = DNN$parameters$`0.weight`,
#       init_bias = DNN$parameters$`0.bias`,
#       init_alpha = 0.9,
#       clip_var = TRUE
#     )
#     
#     self$fc2 = torch_hs(
#       in_features = sim_params$d_hidden1,
#       out_features = sim_params$d_hidden2,
#       use_cuda = FALSE,
#       tau = 1,
#       init_weight = DNN$parameters$`2.weight`,
#       init_bias = DNN$parameters$`2.bias`,
#       init_alpha = 0.9,
#       clip_var = TRUE
#     )
#     
#     self$fc3 = torch_hs(
#       in_features = sim_params$d_hidden2,
#       out_features = sim_params$d_out,
#       use_cuda = FALSE,
#       tau = 1,
#       init_weight = DNN$parameters$`4.weight`,
#       init_bias = DNN$parameters$`4.bias`,
#       init_alpha = 0.9,
#       clip_var = TRUE
#     )
#     
#   },
#   
#   forward = function(x) {
#     x %>%
#       self$fc1() %>%
#       nnf_relu() %>%
#       self$fc2() %>%
#       nnf_relu() %>%
#       self$fc3()
#   },
#   
#   
#   
#   get_model_kld = function(){
#     kl1 = self$fc1$get_kl()
#     kl2 = self$fc2$get_kl()
#     kl3 = self$fc3$get_kl()
#     kld = kl1 + kl2 + kl3
#     return(kld)
#   }
# )
# 
# 
# 
# res <- sim_fcn_hshoe_fcnaldata(
#   sim_ind = 1,
#   sim_params = sim_params,
#   nn_model = MLHS,
#   train_epochs = 1E6, # sim_params$train_epochs,
#   verbose = TRUE,
#   display_alpha_thresh = sim_params$wald_thresh,
#   report_every = 1E3, # sim_params$report_every,
#   want_plots = FALSE,
#   want_fcn_plots = TRUE,
#   save_mod = TRUE,
#   stop_k = 100,
#   stop_streak = 25,
#   burn_in = 5E5
# )



# res <- lapply(
#   1:sim_params$n_sims, 
#   function(X) sim_fcn_hshoe_fcnaldata(
#     sim_ind = X, 
#     sim_params = sim_params,
#     nn_model = MLHS,
#     train_epochs = sim_params$train_epochs,
#     verbose = TRUE,
#     display_alpha_thresh = sim_params$wald_thresh,
#     report_every = sim_params$report_every,
#     want_plots = FALSE,
#     want_fcn_plots = TRUE,
#     save_mod = TRUE
#   )
# )
# 
# # set each simulation result unique name
# res <- setNames(res, paste0("sim_", 1:length(res)))
# 
# 
# contents <- list(
#   "res" = res, 
#   "sim_params" = sim_params
# )
# 
# save(contents, file = here::here("sims", "results", save_fname))
# 
# 














