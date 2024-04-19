##################################################
## Project:   MODULES
## Date:      Apr 19, 2024
## Author:    Arnie Seong
##################################################

# https://torch.mlverse.org/technical/modules/



# # # # # # # # # # # # # # # # # # # # # # # # #
## BASE MODULES ("LAYERS") ----
# # # # # # # # # # # # # # # # # # # # # # # # #

library(torch)
# instantiate layer expecting 3 inputs, returns 1 output
l <- nn_linear(3, 1)
l

# has 2 parameters - weight and bias
l$parameters

# Modules are callable; calling a module executes its forward() method ---
# For a linear layer, matrix-multiplies input and weights, and adds the bias.


data <- torch_randn(10, 3)
data
out <- l(data)
out

# numerically equivalent
out2 <- as_array(data) %*% t(as_array(l$parameters$weight)) + l$parameters$bias
cbind(as_array(out), as_array(out2))

# out has gradient info
out$grad_fn

# haven't called backward() yet, so these don't have gradients
l$weight$grad
l$bias$grad

# so, call backward()
# BUT: error!  b/c backward() expects a scalar tensor (e.g. loss)
# out$backward() 

# so make output a scalar tensor by taking avg
d_avg_d_out <- torch_tensor(10)$`repeat`(10)$unsqueeze(1)$t() 
# I think this is wrong.  Shouldn't it be torch_tensor(1/10)?
out$backward(gradient = d_avg_d_out)
l$weight$grad
l$bias$grad



# # # # # # # # # # # # # # # # # # # # # # # # #
## CHECKING d_avg_d_out ----
# # # # # # # # # # # # # # # # # # # # # # # # #
# try to get same output:
torch_manual_seed(314)
data <- torch_randn(10, 3)
l <- nn_linear(3, 1)
out <- l(data)
d_avg_d_out <- torch_tensor(1/10)$`repeat`(10)$unsqueeze(1)$t() 
out$backward(gradient = d_avg_d_out)
lw_grad <- l$weight$grad
lb_grad <- l$bias$grad


# do over, but run backward() on scout
torch_manual_seed(314)
data <- torch_randn(10, 3)
l2 <- nn_linear(3, 1)
out2 <- l2(data)
scout <- out2$mean()
scout$backward()
l2$weight$grad
l2$bias$grad
lw_grad
lb_grad

# yes, should be 1/10



# # # # # # # # # # # # # # # # # # # # # # # # #
## CONTAINER MODULES ("MODELS") ----
# # # # # # # # # # # # # # # # # # # # # # # # #

model <- nn_sequential(
  nn_linear(3, 16),
  nn_relu(),
  nn_linear(16, 1)
)

model$parameters

model[[1]]$bias
out <- model(data)
out$backward(gradient = torch_tensor(1/10)$`repeat`(10)$unsqueeze(1)$t())
model[[1]]$bias$grad

# place on GPU
model$cuda()
model[[1]]$bias$grad


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

learning_rate <- 1e-4

### training loop --------------------------------------------------------------

for (t in 1:200) {
  
  ### -------- Forward pass -------- 
  
  y_pred <- model(x)
  
  ### -------- compute loss -------- 
  loss <- (y_pred - y)$pow(2)$sum()
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation -------- 
  
  # Zero the gradients before running the backward pass.
  model$zero_grad()
  
  # compute gradient of the loss w.r.t. all learnable parameters of the model
  loss$backward()
  
  ### -------- Update weights -------- 
  
  # Wrap in with_no_grad() because this is a part we DON'T want to record
  # for automatic gradient computation
  # Update each parameter by its `grad`
  
  with_no_grad({
    model$parameters %>% purrr::walk(function(param) param$sub_(learning_rate * param$grad))
  })
  
}

