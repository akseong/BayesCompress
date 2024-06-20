##################################################
## Project:   DEBUGGING CHATGPT conversion from pytorch to R
## Date:      Jun 11, 2024
## Author:    Arnie Seong
##################################################

library(ggplot)
library(torch)
library(here)
source(here("Rcode", "BayesianLayers.R"))

# generate toy data.  First 2 covariates in X have an effect.  
# input dimensionality (# of features)
d_in <- 5

# output dimensionality (response dim)
d_out <- 3
n <- 100
x <- torch_randn(n, d_in)
w1_true <- torch_randn(size = c(d_in, d_out))
w1_true[3:5, ] <- 0
b1_true <- torch_randint(-3, 5, size = c(1, d_out))$'repeat'(c(n, 1))
epsilon <- torch_randn(size = c(n, d_out))
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



# construct dataloader
datloader <- torch::dataset(
  
  name = "linear_data",
  
  initialize = function(x, y) {
    
    self$x <- torch_tensor(x)
    self$y <- torch_tensor(y)
    
  },
  
  .getitem = function(i) {
    
    if (length(self$y$size()) == 1) {
      list(x = self$x[i, ], y = self$y[i])
    } else {
      list(x = self$x[i, ], y = self$y[i, ])
    }
  },
  
  .length = function() {
    self$y$size()[[1]]
  }
)


# 80-20 train-validation split
valid_indices <- sample(1:nrow(x), floor(nrow(x)/5))
train_indices <- setdiff(1:nrow(x), valid_indices)
train_dat <- datloader(x[train_indices, ], y_scalar[train_indices])
valid_dat <- datloader(x[valid_indices, ], y_scalar[valid_indices])

# construct NN
in_features <- train_dat$x$size()[2]
out_features <- length(valid_dat$y[1])

train_dl <- torch::dataloader(train_dat, batch_size = 64, shuffle = TRUE)
batch <- torch::dataloader_make_iter(train_dl) %>% torch::dataloader_next()

d_hidden <- 12

mask <- torch_ones(d_in)
net <- nn_module(
  "testnet",

  initialize <- function() {
    
    self$fc1 <- LinearGroupNJ(
      in_features = in_features, 
      out_features = d_hidden,
      cuda = FALSE, init_weight = NULL, 
      init_bias = NULL, clip_var = NULL)
    
    
    self$fc2 <- LinearGroupNJ(
      in_features = d_hidden, 
      out_features = out_features,
      cuda = FALSE, init_weight = NULL, 
      init_bias = NULL, clip_var = NULL)
    
    self$relu <- nnf_relu()
    
    self$kl_list <- list(self$fc1, self$fc2)
  },

  forward <- function(x) {
    
    x %>% 
      self$fc1() %>%
      self$relu() %>% 
      self$fc2()
  },

  kl_divergence <- function() {
    KLD = 0
    for (layer in self$kl_list){
      KLD = KLD + layer$kl_divergence()
    }
  }
)


model <- net()
model(batch$x)




fc1 <- LinearGroupNJ(
  in_features = in_features, 
  out_features = d_hidden,
  init_weight = NULL, 
  init_bias = NULL, clip_var = NULL)











