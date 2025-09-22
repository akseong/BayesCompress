
#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)

library(torch)
source(here("Rcode", "torch_horseshoe.R"))
source(here("Rcode", "sim_functions.R"))


# datagen fcns ----
fcn1 <- function(x) exp(x/2)
fcn2 <- function(x) cos(pi*x) + sin(pi/1.2*x)
fcn3 <- function(x) abs(x)^(1.5)
fcn4 <- function(x) - (abs(x))
flist = list(fcn1, fcn2, fcn3, fcn4)
plot_datagen_fcns(flist)

# set seed & generate data ----
sim_seed <- 314
  
# simulate same dataset via seed----
n_obs <- 1e4
d_in <- 104
err_sig <- 1
use_cuda <- FALSE

set.seed(sim_seed)
torch_manual_seed(sim_seed)
simdat <- sim_func_data(
  n_obs = n_obs,
  d_in = d_in,
  flist = flist,
  err_sigma = err_sig,
  use_cuda = use_cuda
)



# for plotting
pred_mats <- make_pred_mats(
  flist = flist,
  xgrid = seq(-3, 3, length.out = 100),
  d_in = d_in
)



# initialize regular DNN to get starting weights ----
d_out <- 1
d_hidden1 <- 16
d_hidden2 <- 8
d_hidden3 <- 8
d_hidden4 <- 16
# d_hidden5 <- 8
# d_hidden6 <- 8

DNN <- nn_sequential(
  nn_linear(d_in, d_hidden1),
  nn_relu(),
  nn_linear(d_hidden1, d_hidden2),
  nn_relu(),
  nn_linear(d_hidden2, d_hidden3),
  nn_relu(),
  nn_linear(d_hidden3, d_hidden4),
  nn_relu(),
  nn_linear(d_hidden4, d_out)
)
if (use_cuda) {DNN$to(device = "cuda")}


learning_rate <- 0.08
optimizer <- optim_adam(DNN$parameters, lr = learning_rate)

report_every <- 100
for (t in 1:2500) {
  y_pred <- DNN(simdat$x)
  loss <- nnf_mse_loss(y_pred, simdat$y)
  if (t %% report_every == 0) {
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
    cat("gradient_info, true vars: ", mean(DNN$parameters$`0.weight`$grad[, 1:4])$item(), "\n")
    cat("gradient_info, nuis vars: ", mean(DNN$parameters$`0.weight`$grad[, 5:d_in])$item(), "\n\n")
    # plot true and pred functions:
    plt <- plot_fcn_preds(
      torchmod = DNN,
      pred_mats = pred_mats
    ) + 
      labs(
        title = "predicted (solid) and original (dotted) functions",
        subtitle = paste0("epoch: ", t)
      )
    print(plt)
  } # END UPDATE LOOP
  
  # gradient step
  optimizer$zero_grad()
  loss$backward()
  optimizer$step()
} # END TRAINING LOOP

# results
# When no nuisance vars and 10k obs
# small (dh_1 = 16, dh_2 = 8) regular DNN learns fcns 1:4 fairly well, 
# very quickly (~ 5000 epochs)

# when only a few nuisance vars (2-20) but many obs
# still does well with little training.

# many nuisance vars (100), starts taking longer, 
# doesn't learn functions 2 or 4 (cos wave, negative abs)



























































