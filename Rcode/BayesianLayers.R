##################################################
## Project:   Implementing BayesianLayers
## Date:      Apr 19, 2024
## Author:    Arnie Seong
##################################################

# https://github.com/KarenUllrich/Tutorial_BayesianCompressionForDL/blob/master/BayesianLayers.py

library(torch)

cuda <- cuda_is_available()


reparametrize <- function(mu, logvar, cuda = FALSE, sampling = TRUE) {
  if (sampling) {
    std <- logvar$mul(0.5)$exp_()
    if (cuda) {
      eps <- torch_randn(std$size(), device = "gpu", requires_grad = TRUE)
    } else {
      eps <- torch_randn(std$size(), device = "cpu", requires_grad = TRUE)
    }
    return(mu + eps * std)
  } else {
    return(mu)
  }
}

# CHATGPT generated
# reparametrize <- function(mu, logvar, cuda = FALSE, sampling = TRUE) {
#   if (sampling) {
#     std <- logvar$mul(0.5)$exp_()
#     if (cuda) {
#       eps <- torch$cuda$FloatTensor(std$size())$normal_()
#     } else {
#       eps <- torch$FloatTensor(std$size())$normal_()
#     }
#     eps <- Variable(eps)
#     return(mu + eps * std)
#   } else {
#     return(mu)
#   }
# }




LinearGroupNJ <- torch::nn_module(
  classname = "LinearGroupNJ",
  
  
  public = list(
    initialize = function(in_features, out_features, 
                          cuda = FALSE, 
                          init_weight = NULL, init_bias = NULL, 
                          clip_var = NULL) {
      super$initialize()
      self$cuda <- cuda
      self$in_features <- in_features
      self$out_features <- out_features
      self$clip_var <- clip_var
      self$deterministic <- FALSE
      
      # Trainable parameters according to Eq.(6)
      # Dropout parameters
      self$z_mu <- Parameter(torch$Tensor(in_features))
      self$z_logvar <- Parameter(torch$Tensor(in_features))  # = z_mu^2 * alpha
      # Weight parameters
      self$weight_mu <- Parameter(torch$Tensor(out_features, in_features))
      self$weight_logvar <- Parameter(torch$Tensor(out_features, in_features))
      self$bias_mu <- Parameter(torch$Tensor(out_features))
      self$bias_logvar <- Parameter(torch$Tensor(out_features))
      
      # Initialize parameters either randomly or with pretrained net
      self$reset_parameters(init_weight, init_bias)
      
      # Activations for KL
      self$sigmoid <- nn$Sigmoid()
      self$softplus <- nn$Softplus()
      
      # Numerical stability param
      self$epsilon <- 1e-8
    },
    
    reset_parameters = function(init_weight, init_bias) {
      # Init means
      stdv <- 1 / sqrt(self$weight_mu$size(1))
      self$z_mu$data <- torch$normal(1, 1e-2)
      if (!is.null(init_weight)) {
        self$weight_mu$data <- torch$Tensor(init_weight)
      } else {
        self$weight_mu$data <- torch$normal(0, stdv)
      }
      if (!is.null(init_bias)) {
        self$bias_mu$data <- torch$Tensor(init_bias)
      } else {
        self$bias_mu$data <- torch$zeros(self$out_features)
      }
      
      # Init logvars
      self$z_logvar$data <- torch$normal(-9, 1e-2)
      self$weight_logvar$data <- torch$normal(-9, 1e-2)
      self$bias_logvar$data <- torch$normal(-9, 1e-2)
    },
    
    clip_variances = function() {
      if (!is.null(self$clip_var)) {
        self$weight_logvar$data <- self$weight_logvar$data$clamp(max = log(self$clip_var))
        self$bias_logvar$data <- self$bias_logvar$data$clamp(max = log(self$clip_var))
      }
    },
    
    get_log_dropout_rates = function() {
      log_alpha <- self$z_logvar - torch$log(self$z_mu$pow(2) + self$epsilon)
      return(log_alpha)
    },
    
    compute_posterior_params = function() {
      weight_var <- self$weight_logvar$exp()
      z_var <- self$z_logvar$exp()
      self$post_weight_var <- self$z_mu$pow(2) * weight_var + z_var * self$weight_mu$pow(2) + z_var * weight_var
      self$post_weight_mu <- self$weight_mu * self$z_mu
      return(list(self$post_weight_mu, self$post_weight_var))
    },
    
    forward = function(x) {
      if (self$deterministic) {
        stop("Flag deterministic is True. This should not be used in training.")
      }
      
      
      
      
      
      
      
      ####**** PROBLEMS HERE ****
      
      
      
      
      batch_size <- x$size()[1]
      # Compute z  
      # Note that we reparametrise according to [2] Eq. (11) (not [1])
      z <- reparametrize(
        mu = self$z_mu$`repeat`(c(batch_size, 1)), 
        logvar = self$z_logvar$`repeat`(c(batch_size, 1)), 
        sampling = self$training, 
        cuda = self$cuda
      )
      
      # Apply local reparametrisation trick see [1] Eq. (6)
      # To the parametrisation given in [3] Eq. (6)
      xz <- x * z
      mu_activations <- nnf_linear(xz, self$weight_mu, self$bias_mu)
      var_activations <- nnf_linear(xz$pow(2), self$weight_logvar$exp(), self$bias_logvar$exp())
      
      
      
      
      
      return(reparametrize(mu_activations, var_activations$log(), sampling = self$training, cuda = self$cuda))
    },
    
    kl_divergence = function() {
      # KL(q(z)||p(z))
      # We use the KL divergence approximation given by [2] Eq.(14)
      k1 <- 0.63576
      k2 <- 1.87320
      k3 <- 1.48695
      log_alpha <- self$get_log_dropout_rates()
      KLD <- -torch$sum(k1 * self$sigmoid(k2 + k3 * log_alpha) - 0.5 * self$softplus(-log_alpha) - k1)
      
      # KL(q(w|z)||p(w|z))
      # We use the KL divergence given by [3] Eq.(8)
      KLD_element <- -0.5 * self$weight_logvar + 0.5 * (self$weight_logvar$exp() + self$weight_mu$pow(2)) - 0.5
      KLD <- KLD + torch$sum(KLD_element)
      
      # KL bias
      KLD_element <- -0.5 * self$bias_logvar + 0.5 * (self$bias_logvar$exp() + self$bias_mu$pow(2)) - 0.5
      KLD <- KLD + torch$sum(KLD_element)
      
      return(KLD)
    },
    
    clone = function() {
      new_obj <- super$clone()
      new_obj$z_mu <- self$z_mu$clone()
      new_obj$z_logvar <- self$z_logvar$clone()
      new_obj$weight_mu <- self$weight_mu$clone()
      new_obj$weight_logvar <- self$weight_logvar$clone()
      new_obj$bias_mu <- self$bias_mu$clone()
      new_obj$bias_logvar <- self$bias_logvar$clone()
      return(new_obj)
    }
  )
)










