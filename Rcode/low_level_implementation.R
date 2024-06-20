##################################################
## Project:   low-level Bayesian Compression implementation
## Start:     Jun 17, 2024
## Author:    Arnie Seong
##################################################

library(ggplot2)
library(torch)
library(here)
source(here("Rcode", "BayesianLayers.R"))

# generate toy data.  Only first 2 covariates in X have an effect.  
# input dimensionality (# of features)
d_in <- 5

# output dimensionality (response dim)
datagen_out <- 3
n <- 100
x <- torch_randn(n, d_in)
w1_true <- torch_randn(size = c(d_in, datagen_out))
w1_true[3:5, ] <- 0
b1_true <- torch_randint(-3, 5, size = c(1, datagen_out))$'repeat'(c(n, 1))
epsilon <- torch_randn(size = c(n, datagen_out))
# plot(density(as_array(epsilon)))
# qqnorm(y = as_array(epsilon))

# multivariate response
y_vector <- x$mm(w1_true) + b1_true + epsilon
# scalar response
y_scalar <- y_vector$sum(dim = 2) + epsilon$sum(dim = 2)


lmfit <- lm(as_array(y_scalar) ~ as_array(x))
summary(lmfit)
b1_true[1, ]$sum()
w1_true$sum(dim = 2)
# lm does OK.  tends to miss covars when coefficient < .5  
# (scalar response has noise with variance ~ 3) 




# scalar response
y <- y_scalar$unsqueeze(2)
d_hidden <- 12
d_out <- length(y[1])




# # # # # # # # # # # # # # # # # # # # # # # # #
## LOG UNIFORM MODEL ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# p(z_lj) \propto |z_lj|^{-1}




w1 <- torch_randn(c(d_in, d_hidden), requires_grad = TRUE)
w2 <- torch_randn(c(d_hidden, d_out), requires_grad = TRUE)

b1 <- torch_randn(1, d_hidden, requires_grad = TRUE)
b2 <- torch_randn(1, d_out, requires_grad = TRUE)

z1_mu <- torch_randn(d_in, requires_grad = TRUE)$clamp(min = 0, max = 1)
z1_logvar <- torch_randn(d_in, requires_grad = TRUE)$abs

z2_mu <- torch_randn(d_hidden, requires_grad = TRUE)$clamp(min = 0, max = 1)
z2_logvar <- torch_randn(d_hidden, requires_grad = TRUE)$abs

w1_mu <- torch_randn(c(d_hidden, d_in))
w1_logvar <- torch_randn(c(d_hidden, d_in))

w2_mu <- torch_randn(c(d_out, d_hidden))
w2_logvar <- torch_randn(c(d_out, d_hidden))


b1_mu <- 








learning_rate <- 1e-4



# # # # # # # # # # # # 
##    training loop    
  
  
  # # # # # # # # # # # # 
  ##    Forward pass
  h <- x$mm(w1) + b1
  h_relu <- h$clamp(min = 0)
  y_pred <- h_relu$mm(w2) + b2
  
  
  # # # # # # # # # # # # 
  ##    compute loss
  KL <- 
  loss <- nnf_mse_loss(y_pred, y, reduction = "sum")
  KL
  
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  # # # # # # # # # # # # 
  ##    backprop
  
  
  
  
  # # # # # # # # # # # # 
  ##    update weights    
  
  # Wrap in with_no_grad() because this is a part we DON'T 
  # want to record for automatic gradient computation
  with_no_grad({
    w1 <- w1$sub_(learning_rate * w1$grad)
    w2 <- w2$sub_(learning_rate * w2$grad)
    b1 <- b1$sub_(learning_rate * b1$grad)
    b2 <- b2$sub_(learning_rate * b2$grad)  
    
    # Zero gradients after every pass
    w1$grad$zero_()
    w2$grad$zero_()
    b1$grad$zero_()
    b2$grad$zero_()  
  })
  
  
  
  
  
  








