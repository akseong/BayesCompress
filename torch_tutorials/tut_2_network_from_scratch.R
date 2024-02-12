##################################################
## Project:   TORCH TUTORIAL: NETWORK FROM SCRATCH
## Date:      Feb 12, 2024
## Author:    Arnie Seong
##################################################

# https://torch.mlverse.org/technical/network/


# # # # # # # # # # # # # # # # # # # # # # # # #
## GENERATE DATA ----
# # # # # # # # # # # # # # # # # # # # # # # # #

library(torch)
torch_randn(2, 3, 4)

# input dimensionality (# of features)
d_in <- 3

# output dimensionality (response dim)
d_out <- 1

n <- 100
x <- torch_randn(n, d_in)
# torch_randn puts draws from N(0,1) into a tensor
y <- x[, 1, drop = FALSE] * 0.2 - x[, 2, drop = FALSE] * 1.3 - x[, 3, drop = FALSE] * 0.5 + torch_randn(n, 1)


# # # # # # # # # # # # # # # # # # # # # # # # #
## INITIALIZE WEIGHT TENSORS ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# dimensionality of hidden layer
d_hidden <- 32

# weights connecting input to hidden layer
w1 <- torch_randn(d_in, d_hidden)
# weights connecting hidden to output layer
w2 <- torch_randn(d_hidden, d_out)

# hidden layer bias
b1 <- torch_zeros(1, d_hidden)
# output layer bias
b2 <- torch_zeros(1, d_out)



# # # # # # # # # # # # # # # # # # # # # # # # #
## TRAINING LOOP ----
# # # # # # # # # # # # # # # # # # # # # # # # #

train_1_hidden_layer_network <- function(x, y, d_hidden = 32, epochs = 20, learning_rate = 1e-4){
  d_in <- dim(x)[2]
  d_out <- dim(y)[2]
  
  # # # # # # # # # # # # 
  ##    INITIALIZE WEIGHTS/BIASES
  
  # weights connecting input to hidden layer
  w1 <- torch_randn(d_in, d_hidden)
  # weights connecting hidden to output layer
  w2 <- torch_randn(d_hidden, d_out)
  
  # hidden layer bias
  b1 <- torch_zeros(1, d_hidden)
  # output layer bias
  b2 <- torch_zeros(1, d_out)
  
  train_loss <- rep(NA, epochs)
  for (n_epoch in 1:epochs){
    
    # # # # # # # # # # # # 
    ##    FORWARD PASS
    
    # compute pre-activations of hidden layers (dim: 100 x 32)
    # torch_mm does matrix multiplication (`$mm(w1)` calls `torch_mm(self, w1)`)
    h <- x$mm(w1) + b1
    
    # a <- x$mm(w1)
    # b <- torch_mm(x, w1)
    # all.equal(a, b) # TRUE
    
    # ReLU activation function (dim: 100 x 32)
    # torch_clamp cuts off values below/above given thresholds
    h_relu <- h$clamp(min = 0)
    
    # compute output (dim: 100 x 1)
    y_pred <- h_relu$mm(w2) + b2
    
    
    # # # # # # # # # # # # 
    ##    COMPUTE LOSS
    loss <- as.numeric((y_pred - y)$pow(2)$sum())
    train_loss[n_epoch] <- loss
    
    cat(paste0("training loss for epoch ", n_epoch, ": ", round(loss, 5), " \n "))
    
    # # # # # # # # # # # # 
    ##    BACKWARD PASS (BACKPROP)    
    
    # gradient of loss w.r.t. prediction (dim: 100 x 1)
    grad_y_pred <- 2 * (y_pred - y)
    # gradient of loss w.r.t. w2 (dim: 32 x 1)
    grad_w2 <- h_relu$t()$mm(grad_y_pred)
    # gradient of loss w.r.t. hidden activation (dim: 100 x 32)
    grad_h_relu <- grad_y_pred$mm(w2$t())
    # gradient of loss w.r.t. hidden pre-activation (dim: 100 x 32)
    grad_h <- grad_h_relu$clone()
    
    grad_h[h < 0] <- 0
    
    # gradient of loss w.r.t. b2 (shape: ())
    grad_b2 <- grad_y_pred$sum()
    
    # gradient of loss w.r.t. w1 (dim: 3 x 32)
    grad_w1 <- x$t()$mm(grad_h)
    # gradient of loss w.r.t. b1 (shape: (32, ))
    grad_b1 <- grad_h$sum(dim = 1)
    
    
    # # # # # # # # # # # # 
    ##    UPDATE WEIGHTS
    
    if (n_epoch < epochs){
      w2 <- w2 - learning_rate * grad_w2
      b2 <- b2 - learning_rate * grad_b2
      w1 <- w1 - learning_rate * grad_w1
      b1 <- b1 - learning_rate * grad_b1
    }
  }  
  
  return( 
    list(
      "w1" = w1,
      "b1" = b1,
      "w2" = w2,
      "b2" = b2,
      "y_pred" = y_pred
    )
  )
}


net <- train_1_hidden_layer_network(x, y, d_hidden = 25, epochs = 1000, learning_rate = 1e-4)
sum((as.numeric(net$y_pred) - ymat)^2)

net_pred <- function(new_x, net){
  h <- new_x$mm(net$w1) + net$b1
  # a <- x$mm(w1)
  # b <- torch_mm(x, w1)
  # all.equal(a, b) # TRUE
  
  # ReLU activation function (dim: 100 x 32)
  # torch_clamp cuts off values below/above given thresholds
  h_relu <- h$clamp(min = 0)
  
  # compute output (dim: 100 x 1)
  y_pred <- h_relu$mm(net$w2) + net$b2
  
  return(y_pred)
}

gen_dat <- function(n=100, d_in=3){
  x <- torch_randn(n, d_in)
  # torch_randn puts draws from N(0,1) into a tensor
  y <- x[, 1, drop = FALSE] * 0.2 - x[, 2, drop = FALSE] * 1.3 - x[, 3, drop = FALSE] * 0.5 + torch_randn(n, 1)
  return(list("x" = x, "y" = y))  
}

new_dat <- gen_dat(n=100, d_in = 3)
new_y_pred <- net_pred(new_x = new_dat$x, net = net)
ssy <- as.numeric(sum((new_y_pred - new_dat$y)^2))
ssy

# comparison to lm()
ymat <- as.numeric(y)
xmat <- matrix(as.numeric(x), ncol = 3)

# check
xmat[1:3,]
x[1:3,]

lm_mod <- lm(ymat ~ xmat)
summary(lm_mod)

sum((lm_mod$fitted - ymat)^2)
lm_y_pred <- predict(lm_mod, newdata = as.data.frame(matrix(as.numeric(new_dat$x), ncol = 3)))
sum((lm_y_pred - as.numeric(new_dat$y))^2)


