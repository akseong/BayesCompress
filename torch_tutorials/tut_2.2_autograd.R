##################################################
## Project:   AUTOGRAD
## Date:      Apr 17, 2024
## Author:    Arnie Seong
##################################################

#https://torch.mlverse.org/technical/autograd/

# # # # # # # # # # # # # # # # # # # # # # # # #
## AUTOGRAD basics ----
# # # # # # # # # # # # # # # # # # # # # # # # #

library(torch)

# torch uses a module called autograd to
# (1) record operations performed on tensors, and
# (2) store what will have to be done to obtain the corresponding gradients, 
# once we’re entering the backward pass.
# These prospective actions are stored internally as functions, and when 
# it’s time to compute the gradients, these functions are applied in order: 
# Application starts from the output node, and calculated gradients are 
# successively propagated back through the network. This is a form of reverse 
# mode automatic differentiation

# MUST ENABLE autograd operations on a tensor
# x is now a tensor with respect to which gradients have to be calculated --- usually
# a tensor representing weights/biases, NOT the input data.
x <- torch_ones(2, 2, requires_grad = TRUE)

# now, y inherits a gradient computation operation
y <- x$mean()
y
y$grad_fn

# actual computation of gradient is triggered by 
# calling backward()) on the OUTPUT tensor
y$backward()

# now x has a non-null field grad that stores 
# the gradient of Y w.r.t. X
x$grad



# # # # # # # # # # # # # # # # # # # # # # # # #
## EXAMPLE ----
# # # # # # # # # # # # # # # # # # # # # # # # #

x1 <- torch_ones(2, 2, requires_grad = TRUE)
x2 <- torch_tensor(1.1, requires_grad = TRUE)

y <- x1 * (x2 + 2)
z <- y$pow(2) * 3

out <- z$mean()

# intermediate gradience usually not stored
# to store, call retain_grad()
y$retain_grad()
z$retain_grad()

# how to compute the gradient for mean, the last operation executed
out$grad_fn

# how to compute the gradient for the multiplication by 3 in z = y.pow(2) * 3
out$grad_fn$next_functions

# how to compute the gradient for pow in z = y.pow(2) * 3
out$grad_fn$next_functions[[1]]$next_functions

# how to compute the gradient for the multiplication in y = x * (x + 2)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions

# how to compute the gradient for the two branches of y = x * (x + 2),
# where the left branch is a leaf node (AccumulateGrad for x1)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions[[1]]$next_functions

# here we arrive at the other leaf node (AccumulateGrad for x2)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions[[1]]$next_functions[[2]]$next_functions


# now, compute gradients by calling backward()
# computes gradients for all tensors in path
out$backward()

z$grad
y$grad
x2$grad
x1$grad




# # # # # # # # # # # # # # # # # # # # # # # # #
## simple network using autograd ----
# # # # # # # # # # # # # # # # # # # # # # # # #

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


### initialize weights ---------------------------------------------------------

# dimensionality of hidden layer
d_hidden <- 32
# weights connecting input to hidden layer
w1 <- torch_randn(d_in, d_hidden, requires_grad = TRUE)
# weights connecting hidden to output layer
w2 <- torch_randn(d_hidden, d_out, requires_grad = TRUE)

# hidden layer bias
b1 <- torch_zeros(1, d_hidden, requires_grad = TRUE)
# output layer bias
b2 <- torch_zeros(1, d_out, requires_grad = TRUE)

### network parameters ---------------------------------------------------------

learning_rate <- 1e-4

### training loop --------------------------------------------------------------

for (t in 1:200) {
  ### -------- Forward pass --------
  # next line does the matrix ops for the first layer, relu ($clamp), matrix ops for 2nd layer
  y_pred <- x$mm(w1)$add(b1)$clamp(min = 0)$mm(w2)$add(b2)
  
  ### -------- compute loss -------- 
  loss <- (y_pred - y)$pow(2)$sum()
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation --------
  
  # compute gradient of loss w.r.t. all tensors with requires_grad = TRUE
  loss$backward()
  
  ### -------- Update weights -------- 
  
  # Wrap in with_no_grad() because this is a part we DON'T 
  # want to record for automatic gradient computation
  with_no_grad({
    w1 <- w1$sub_(learning_rate * w1$grad)
    w2 <- w2$sub_(learning_rate * w2$grad)
    b1 <- b1$sub_(learning_rate * b1$grad)
    b2 <- b2$sub_(learning_rate * b2$grad)  
    
    # Zero gradients after every pass, as they'd accumulate otherwise
    w1$grad$zero_()
    w2$grad$zero_()
    b1$grad$zero_()
    b2$grad$zero_()  
  })
  
}





