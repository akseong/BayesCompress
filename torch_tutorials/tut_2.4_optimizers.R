##################################################
## Project:   OPTIMIZERS
## Date:      Apr 19, 2024
## Author:    Arnie Seong
##################################################

# https://torch.mlverse.org/technical/optimizers/


# # # # # # # # # # # # # # # # # # # # # # # # #
## LOSS FUNCTIONS ----
# # # # # # # # # # # # # # # # # # # # # # # # #

library(torch)
x <- torch_randn(c(3,2,3))
y <- torch_randn(c(3,2,3))

# mean squared error
nnf_mse_loss(x,y)

# loss functions to call directly start with nnf_
# nnf_binary_cross_entropy()
# nnf_nll_loss()   # neg log like
# nnf_kl_div()

# can also define and call later using nn_
loss <- nn_mse_loss()
loss(x, y)


# # # # # # # # # # # # # # # # # # # # # # # # #
## OPTIMIZERS ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# using optim_adam() --- ADAM algorithm --- instead of basic gradient descent

data <- torch_randn(1, 3)
model <- nn_linear(3, 1)
model$parameters

optimizer <- optim_adam(model$parameters, lr = 0.01)
optimizer

optimizer$param_groups[[1]]$params

out <- model(data)
out$backward()

# these are still the same as they were
optimizer$param_groups[[1]]$params
model$parameters

# calling step() on the optimizer PERFORMS the updates
optimizer$step()

# now they have been updated
optimizer$param_groups[[1]]$params
model$parameters

# if calling in a loop, must make sure to call zero_grad()
# otherwise gradients will be accumulated
optimizer$zero_grad()






# # # # # # # # # # # # # # # # # # # # # # # # #
## SIMPLE NETWORK WITH OPTIM ----
# # # # # # # # # # # # # # # # # # # # # # # # #
library(torch)

### generate training data -----------------------------------------------------

# input dimensionality (number of input features)
d_in <- 3
# output dimensionality (number of predicted features)
d_out <- 1
# number of observations in training set
n <- 100


# create random data
x <- torch_randn(n, d_in)
y <- x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5 + torch_randn(n, 1)



### define the network ---------------------------------------------------------

# dimensionality of hidden layer
d_hidden <- 32

model <- nn_sequential(
  nn_linear(d_in, d_hidden),
  nn_relu(),
  nn_linear(d_hidden, d_out)
)

### network parameters ---------------------------------------------------------

# for adam, need to choose a much higher learning rate in this problem
learning_rate <- 0.08

optimizer <- optim_adam(model$parameters, lr = learning_rate)

### training loop --------------------------------------------------------------

for (t in 1:200) {
  
  ### -------- Forward pass -------- 
  
  y_pred <- model(x)

  ### -------- compute loss -------- 
  loss <- nnf_mse_loss(y_pred, y, reduction = "sum")
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation -------- 
  
  # Still need to zero out the gradients before the backward pass, only this time,
  # on the optimizer object
  optimizer$zero_grad()
  
  # gradients are still computed on the loss tensor (no change here)
  loss$backward()
  
  ### -------- Update weights -------- 
  
  # use the optimizer to update model parameters
  optimizer$step()
}
