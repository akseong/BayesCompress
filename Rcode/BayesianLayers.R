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



LinearGroupNJ <- torch::nn_module(
  classname = "LinearGroupNJ",
  
    initialize = function(in_features, out_features, 
                          # cuda = FALSE, 
                          init_weight = NULL, init_bias = NULL, 
                          clip_var = NULL) {
      super$initialize()
      # self$cuda <- cuda
      self$in_features <- in_features
      self$out_features <- out_features
      self$clip_var <- clip_var
      self$deterministic <- FALSE
      
      # Trainable parameters according to Eq.(6)
      # Dropout parameters
      self$z_mu <- nn_parameter(torch_empty(in_features))
      self$z_logvar <- nn_parameter(torch_empty(in_features))  # = z_mu^2 * alpha
      # Weight parameters
      self$weight_mu <- nn_parameter(torch_empty(out_features, in_features))
      self$weight_logvar <- nn_parameter(torch_empty(out_features, in_features))
      self$bias_mu <- nn_parameter(torch_empty(out_features))
      self$bias_logvar <- nn_parameter(torch_empty(out_features))
      
      # Initialize parameters either randomly or with pretrained net
      self$reset_parameters(init_weight, init_bias)
      
      # Activations for KL
      self$sigmoid <- nn_sigmoid()
      self$softplus <- nn_softplus()
      
      # Numerical stability param
      self$epsilon <- 1e-8
    },
    
    reset_parameters = function(init_weight, init_bias) {
      # Init means
      stdv <- 1 / sqrt(self$weight_mu$size(1))
      self$z_mu$data <- torch_normal(1, 1e-2)
      if (!is.null(init_weight)) {
        self$weight_mu$data <- torch_tensor(init_weight)
      } else {
        self$weight_mu$data <- torch_normal(0, stdv)
      }
      if (!is.null(init_bias)) {
        self$bias_mu$data <- torch_tensor(init_bias)
      } else {
        self$bias_mu$data <- torch_zeros(self$out_features)
      }
      
      # Init logvars
      self$z_logvar$data <- torch_normal(-9, 1e-2)
      self$weight_logvar$data <- torch_normal(-9, 1e-2)
      self$bias_logvar$data <- torch_normal(-9, 1e-2)
    },
    
    clip_variances = function() {
      if (!is.null(self$clip_var)) {
        self$weight_logvar$data <- self$weight_logvar$data$clamp(max = log(self$clip_var))
        self$bias_logvar$data <- self$bias_logvar$data$clamp(max = log(self$clip_var))
      }
    },
    
    get_log_dropout_rates = function() {
      log_alpha <- self$z_logvar - torch_log(self$z_mu$pow(2) + self$epsilon)
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
      
      batch_size <- x$size()[1]
      # Compute z  
      # Note that we reparametrise according to [2] Eq. (11) (not [1])
      z <- reparametrize(
        mu = self$z_mu$`repeat`(c(batch_size, 1)), 
        logvar = self$z_logvar$`repeat`(c(batch_size, 1)), 
        sampling = self$training
        # , 
        # cuda = self$cuda
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
    }
  # ,
  #   
  #   clone = function() {
  #     new_obj <- super$clone()
  #     new_obj$z_mu <- self$z_mu$clone()
  #     new_obj$z_logvar <- self$z_logvar$clone()
  #     new_obj$weight_mu <- self$weight_mu$clone()
  #     new_obj$weight_logvar <- self$weight_logvar$clone()
  #     new_obj$bias_mu <- self$bias_mu$clone()
  #     new_obj$bias_logvar <- self$bias_logvar$clone()
  #     return(new_obj)
  #   }
  # 
)




# # # # # # # # # # # # # # # # # # # # # # # # #
## COMPRESSION ----
# # # # # # # # # # # # # # # # # # # # # # # # #

# General tools
unit_round_off <- function(t = 23) {
  # Calculate unit round off based on the number of significand bits
  return(0.5 * 2 ** (1 - t))
}

SIGNIFICANT_BIT_PRECISION <- lapply(1:23, function(i) unit_round_off(t = i + 1))

float_precision <- function(x) {
  # Compute float precision based on predefined significant bit precision levels
  sum(sapply(SIGNIFICANT_BIT_PRECISION, function(sbp) x < sbp))
}

float_precisions <- function(X, dist_fun, layer = 1) {
  X <- np$flatten(X)
  out <- sapply(X * 2, float_precision)
  ceiling(dist_fun(out))
}

special_round <- function(input, significant_bit) {
  delta <- unit_round_off(t = significant_bit)
  rounded <- floor(input / delta + 0.5)
  rounded * delta
}

fast_infernce_weights <- function(w, exponent_bit, significant_bit) {
  special_round(w, significant_bit)
}

compress_matrix <- function(x) {
  if (length(np$shape(x)) != 2) {
    dim <- np$shape(x)
    x <- np$reshape(x, c(dim[1] * dim[2], dim[3] * dim[4]))
    x <- x[, np$any(x != 0, axis = 0)]
    x <- x[np$any(x != 0, axis = 1), ]
  } else {
    x <- x[np$any(x != 0, axis = 1), ]
    x <- x[, np$any(x != 0, axis = 0)]
  }
  x
}

extract_pruned_params <- function(layers, masks) {
  post_weight_mus <- list()
  post_weight_vars <- list()
  
  for (i in 1:length(layers)) {
    layer <- layers[[i]]
    mask <- masks[[i]]
    
    # Compute posteriors
    post_weight_mu <- layer$compute_posterior_params()$cpu()$data$numpy()
    post_weight_var <- layer$compute_posterior_params()$cpu()$data$numpy()
    
    # Apply mask to mus and variances
    post_weight_mu <- post_weight_mu * mask
    post_weight_var <- post_weight_var * mask
    
    post_weight_mus[[i]] <- post_weight_mu
    post_weight_vars[[i]] <- post_weight_var
  }
  
  list(post_weight_mus, post_weight_vars)
}

compute_compression_rate <- function(vars, in_precision = 32, dist_fun = function(x) max(x), overflow = 10e38) {
  sizes <- sapply(vars, function(v) np$size(v))
  nb_weights <- sum(sizes)
  IN_BITS <- in_precision * nb_weights
  
  # Prune architecture
  vars <- lapply(vars, function(v) compress_matrix(v))
  sizes <- sapply(vars, function(v) np$size(v))
  
  # Compute significant bits
  significant_bits <- sapply(vars, function(v, k) float_precisions(v, dist_fun, layer = k + 1), simplify = FALSE)
  # problems here
  
  
  
  exponent_bit <- ceiling(log2(log2(overflow) + 1) + 1)
  total_bits <- sapply(significant_bits, function(sb) 1 + exponent_bit + sb)
  
  OUT_BITS <- sum(sizes * total_bits)
  
  list(nb_weights / sum(sizes), IN_BITS / OUT_BITS, significant_bits, exponent_bit)
}

display_compression_rate <- function(layers, masks) {
  # Reduce architecture
  pruned_params <- extract_pruned_params(layers, masks)
  weight_mus <- pruned_params[[1]]
  weight_vars <- pruned_params[[2]]
  
  # Compute overflow level based on maximum weight
  overflow <- max(sapply(weight_mus, function(w) max(abs(w))))
  
  # Compute compression rate
  result <- compute_compression_rate(weight_vars, dist_fun = function(x) mean(x), overflow = overflow)
  cat("Compressing the architecture will decrease the model by a factor of", result[[1]], "\n")
  cat("Making use of weight uncertainty can reduce the model by a factor of", result[[2]], "\n")
}

# compute_reduced_weights <- function(layers, masks) {
#   pruned_params <- extract_pruned_params(layers, masks)
#   weight_mus <- pruned_params[[1]]
#   weight_vars <- pruned_params[[2]]
#   
#   overflow <- max(sapply(weight_mus, function(w) max(abs(w))))
#   
#   _, _, significant_bits, exponent_bits <- _compute_compression_rate(weight_vars, dist_fun = function(x) mean(x), overflow = overflow)
#   
#   weights <- mapply(fast_infernce_weights, weight_mus, exponent_bits, significant_bits)
#   weights
# }

# Example usage:
# Assuming layers and masks are defined appropriately
# compute_compression_rate(layers, masks)
# compute_reduced_weights(layers, masks)





