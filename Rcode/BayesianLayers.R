##################################################
## Project:   Implementing BayesianLayers
## Date:      Apr 19, 2024
## Author:    Arnie Seong
##################################################

# https://github.com/KarenUllrich/Tutorial_BayesianCompressionForDL/blob/master/BayesianLayers.py

library(torch)

reparameterize <- function(mu, logvar, use_cuda = FALSE, sampling = TRUE) {
  # for X ~ N(mu, sigma^2), can rewrite as X = mu + sigma * Z
  # "reparam trick" often used to preserve gradient on mu and sigma
  # Last modified 2024/07/16
  if (sampling) {
    std <- logvar$mul(0.5)$exp_()
  if (use_cuda) {
      eps <- torch_randn(std$size(), device = "gpu", requires_grad = TRUE)
    } else {
      eps <- torch_randn(std$size(), device = "cpu", requires_grad = TRUE)
    }
    return(mu$add(eps$mul(std)))
  } else {
    return(mu)
  }
}


BayesianLayerNJ <- nn_module(
  # last modified 2024/07/16
  classname = "BayesianLayerNJ",
  
  initialize = function(
    in_features, out_features,
    use_cuda = FALSE,
    init_weight = NULL,
    init_bias = NULL,
    clip_var = NULL
  ){
    
    self$use_cuda <- use_cuda
    self$in_features <- in_features
    self$out_features <- out_features
    self$clip_var <- clip_var
    self$deterministic <- FALSE
    
    # trainable parameters
    self$z_mu <- nn_parameter(torch_randn(in_features))
    self$z_logvar <- nn_parameter(torch_randn(in_features))
    self$weight_mu <- nn_parameter(torch_randn(out_features, in_features))
    self$weight_logvar <- nn_parameter(torch_randn(out_features, in_features))
    self$bias_mu <- nn_parameter(torch_randn(out_features))
    self$bias_logvar <- nn_parameter(torch_randn(out_features))
    
    # initialize parameters randomly or with pretrained net
    self$reset_parameters(init_weight, init_bias)
    
    # numerical stability param
    self$epsilon <- 1e-8
  },
  
  
  reset_parameters = function(init_weight, init_bias){
    
    # feel like there may be issues with using nn_parameter here again 
    # to populate each of these, but not sure  
    # how to modify in-place without losing `is_nn_parameter() = TRUE`
    
    # initialize means
    stdv <- 1 / sqrt(self$weight_mu$size(1)) # self$weight_mu$size(1) = out_features
    self$z_mu <- nn_parameter(torch_normal(1, 1e-2, size = self$z_mu$size()))      # potential issue (if not considered leaf node anymore?)  wrap in nn_parameter()?
    if (!is.null(init_weight)) {
      self$weight_mu <- nn_parameter(torch_tensor(init_weight))
    } else {
      self$weight_mu <- nn_parameter(torch_normal(0, stdv, size = self$weight_mu$size()))
    }
    
    if (!is.null(init_bias)) {
      self$bias_mu <- nn_parameter(torch_tensor(init_bias))
    } else {
      self$bias_mu <- nn_parameter(torch_zeros(self$out_features))
    }
    
    # initialize log variances
    self$z_logvar <- nn_parameter(torch_normal(log(1/2), 1e-2, size = self$in_features)) 
    # z_logvar init changed from original proposed init value to make dropout parameter alpha ~ 1/2
    self$weight_logvar <- nn_parameter(torch_normal(-9, 1e-2, size = c(self$out_features, self$in_features)))
    self$bias_logvar <- nn_parameter(torch_normal(-9, 1e-2, size = self$out_features))
  },
  
  
  clip_variances = function() {
    if (!is.null(self$clip_var)) {
      self$weight_logvar <- nn_parameter(self$weight_logvar$clamp(max = log(self$clip_var)))
      self$bias_logvar <- nn_parameter(self$bias_logvar$clamp(max = log(self$clip_var)))
    }
  },
  
  
  get_log_dropout_rates = function() {
    log_alpha = self$z_logvar - torch_log(self$z_mu$pow(2) + self$epsilon)
    return(log_alpha)
  },
  
  
  compute_posterior_param = function() {
    weight_var <- self$weight_logvar$exp()
    z_var <- self$z_logvar$exp()
    self$post_weight_var <- self$z_mu$pow(2) * weight_var + z_var * self$weight_mu$pow(2) + z_var * weight_var
    self$post_weight_mu <- self$weight_mu * self$z_mu
    return(list(
      "post_weight_mu" = self$post_weight_mu,
      "post_weight_var" = self$post_weight_var
    ))
  },
  
  
  forward = function(x){
    if (self$deterministic) {
      cat("argument deterministic = TRUE.  Should not be used for training")
      return(
        nnf_linear(
          input = x, 
          weight = self$post_weight_mu, 
          bias = self$bias_mu
        )
      )
    }
    batch_size <- x$size(1)
    z <- reparameterize(
      mu = self$z_mu$'repeat'(c(batch_size, 1)), 
      logvar = self$z_logvar$'repeat'(c(batch_size, 1)),
      sampling = !self$deterministic,
      use_cuda = self$use_cuda
    )
    xz <- x*z
    mu_activations <- nnf_linear(
      input = xz, 
      weight = self$weight_mu, 
      bias = self$bias_mu
    )
    var_activations <- nnf_linear(
      input = xz$pow(2), 
      weight = self$weight_logvar$exp(), 
      bias = self$bias_logvar$exp()
    )
    
    return(
      reparameterize(
        mu = mu_activations, 
        logvar = var_activations$log(), 
        use_cuda = self$use_cuda, 
        sampling = !self$deterministic
      )
    )
  },
  
  
  get_kl = function() {
    k1 = 0.63576
    k2 = 1.87320
    k3 = 1.48695
    log_alpha = self$get_log_dropout_rates()
    
    # KL(q(z) || p(z))
    kl_z <- -torch_sum(
      k1 * nnf_sigmoid(k2 + k3*log_alpha) - 0.5 * nnf_softplus(-log_alpha) - k1
    )
    
    # KL(q(w|z) || p(w|z))
    kl_w_z <- torch_sum(
      -0.5 * self$weight_logvar + 0.5 * (self$weight_logvar$exp() + self$weight_mu$pow(2)) - 0.5
    )
    
    # KL for bias term
    kl_bias <- torch_sum(
      -0.5 * self$bias_logvar + 0.5 * (self$bias_logvar$exp() + self$bias_mu$pow(2)) - 0.5
    )
    
    # sum
    kl <- kl_z + kl_w_z + kl_bias
    return(kl)
  }
)


