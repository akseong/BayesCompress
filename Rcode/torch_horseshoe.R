##################################################
## Project:   Implementing Horseshoe
## Date:      Jan 3, 2025
## Author:    Arnie Seong
##################################################


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




KL_lognorm_gamma <- function(mu, sig, a = 1/2, b = 1){
  # for s_a, alpha_i
  # (LogNormal q || Gamma p)
  # log(b) - 1/b * (exp(1/2 * sig^2 + mu)) + 1/2 * (mu + 2*log(sig) + 1 + log(2))
  expr2 <- ((mu$add( (sig$pow(2))$mul(1/2) ))$exp())$mul(1/b)
  expr3 <- (mu$add( (sig$log())$mul(2) )$add(1)$add(log(2)))$mul(1/2)
  return(log(b)$add(expr2)$add(expr3))
}

KL_lognorm_IG <- function(mu, sig, a = 1/2, b = 1){
  # for s_b, beta_i
  # i.e. (LogNormal q || Inverse-Gamma p)
  # log(b) - 1/b * exp(1/2 * sig^2 - mu) + 1/2 * (- mu + 2*log(sig) + 1 + log(2))
  KL_lognorm_gamma(-mu, sig, a, b=1)
}



E_lognorm <- function(mu, sig){
  # exp(mu + (sig^2)/2)
  # ln_mu <- mu$add( (sig$pow(2))$mul(0.5) )
  # return(ln_mu$exp())
  # alternatively:
  (mu$add( (sig$pow(2))$mul(0.5) ))$exp()
}


V_lognorm <- function(mu, sig){
  # (exp(sig^2) - 1) * exp(2*mu + sig^2)
  expr1 <- ((sig$pow(2))$exp())$add(-1)
  expr2 <- (mu$mul(2))$add(sig$pow(2))
  return(expr1$mul(expr2$exp()))
  # alternatively
  # sig$pow(2)$exp()$add(-1)$mul(    (mu$mul(2)$add(sig$pow(2)))$exp()   )
}


E_sqrt_lognorm <- function(mu, sig){
  # if X ~ LN(mu, sig), then X = exp(Y) with Y~N(mu, sig)
  # The sqrt(X) = sqrt(exp^Y) = exp^{Y/2} and Y/2 ~ N(mu/2, sig/2)
  E_lognorm(mu/2, sig/2)
}

V_sqrt_lognorm <- function(mu, sig){
  V_lognorm(mu/2, sig/2)
}

V_xy <- function(Ex, Ey, Vx, Vy){
  # Vx*Vy + Vx*(Ey^2) + Vy*(Ex^2)
  Vx$mul(Vy)$add(Vx$mul(Ey$pow(2)))$add(Vy$mul(Ex$pow(2)))
}




## VARIATIONAL DISTRIBUTION PARAMETERS FOR \tilde{z}, s
# z_i = \sqrt(atilde_i btilde_i s_a s_b) = ztilde_i s
## \tilde{z} = \sqrt{\tilde{\alpha}_i \tilde{\beta}_i}    
##   \sim  LogNormal with 
##         mu =     \frac{1}{2}   (\mu_{\tilde{alpha}_i} + \mu_{\tilde{beta}_i} )
##         variance = \frac{1}{4}   (\sigma^2_{\tilde{\alpha}_i} + \sigma^2_{\tilde{\beta}_i}) 
mu_sqrt_prod_LN <- function(mu_1, mu_2){
  # \frac{mu_1 + mu_2}{2}
  (mu_1$add(mu_2))$mul(1/2)
}

logvar_sqrt_prod_LN <- function(logvar_1, logvar_2){
  # \frac{sig^2_1 + sig^2_2}{4}
  ((logvar_1$exp())$add(logvar_2$exp()))$log() - log(4)
}

ztilde_mu <- mu_sqrt_prod_LN(atilde_mu, btilde_mu)
ztilde_logvar <- logvar_sqrt_prod_LN(atilde_logvar, btilde_logvar)

s_mu <- mu_sqrt_prod_LN(atilde_mu, btilde_mu)
s_logvar <- logvar_sqrt_prod_LN(sa_mu, sb_mu)


# z_mu_fcn <- function(sa_mu, sb_mu, atilde_mu, btilde_mu,
#                      sa_logvar, sb_logvar, atilde_logvar, btilde_logvar){
#   # compute mu_z
#   E_sa <- E_lognorm(sa_mu, sa_logvar$exp()) 
#   E_sb <- E_lognorm(sb_mu, sb_logvar$exp())
#   E_at <- E_lognorm(atilde_mu, atilde_logvar$exp())
#   E_bt <- E_lognorm(btilde_mu, btilde_logvar$exp())
#   E_sa$mul(E_sb$mul(E_at$mul(E_bt)))$exp(0.5)
# }
# 
# z_logvar_fcn <- function(sa_mu, sb_mu, atilde_mu, btilde_mu,
#                          sa_logvar, sb_logvar, atilde_logvar, btilde_logvar){
#   # compute variance of Z
#   E_sa <- E_lognorm(sa_mu, sa_logvar$exp()) 
#   E_sb <- E_lognorm(sb_mu, sb_logvar$exp())
#   E_at <- E_lognorm(atilde_mu, atilde_logvar$exp())
#   E_bt <- E_lognorm(btilde_mu, btilde_logvar$exp())
#   
#   V_sa <- V_lognorm()
#   V_sb <- V_lognorm()
#   V_at <- V_lognorm()
#   V_bt <- V_lognorm() 
# }








torch_hs <- nn_module(
  # last modified 1/3/2025
  classname = "horseshoe_layer",
  
  initialize = function(
    in_features, out_features,
    use_cuda = FALSE,
    init_weight = NULL,
    init_bias = NULL,
    init_alpha = 1/2,
    clip_var = NULL
  ){
    
    self$use_cuda <- use_cuda
    self$in_features <- in_features
    self$out_features <- out_features
    self$clip_var <- clip_var
    self$deterministic <- FALSE
    
    #### trainable parameters
    # s = global scal param
    # s^2 = sa*sb
    self$sa_mu <- nn_parameter(torch_randn(1))
    self$sa_logvar <- nn_parameter(torch_randn(1))
    self$sb_mu <- nn_parameter(torch_randn(1))
    self$sb_logvar <- nn_parameter(torch_randn(1))
    # z_i_tilde = local scale param
    # z_i_tilde^2 = alpha_tilde * beta_tilde
    self$atilde_mu <- nn_parameter(torch_randn(in_features))
    self$atilde_logvar <- nn_parameter(torch_randn(in_features))
    self$btilde_mu <- nn_parameter(torch_randn(in_features))
    self$btilde_logvar <- nn_parameter(torch_randn(in_features))
    
    # weight dist'n params
    self$weight_mu <- nn_parameter(torch_randn(out_features, in_features))
    self$weight_logvar <- nn_parameter(torch_randn(out_features, in_features))
    self$bias_mu <- nn_parameter(torch_randn(out_features))
    self$bias_logvar <- nn_parameter(torch_randn(out_features))
    
  
    # composite vars
    # z = \sqrt{  s_a s_b \tilde{\alpha} \tilde{\beta}}
    # s = \sqrt{  s_a s_b  }
    # \tilde{z} = \sqrt{  \tilde{\alpha} \tilde{\beta}  }
    
    
    
    
      
    
    
    # initialize parameters randomly or with pretrained net
    self$reset_parameters(init_weight, init_bias, init_alpha)
    
    # numerical stability param
    self$epsilon <- 1e-8
  },
  
  
  reset_parameters = function(init_weight, init_bias, init_alpha){
    
    # feel like there may be issues with using nn_parameter here again 
    # to populate each of these, but not sure  
    # how to modify in-place without losing `is_nn_parameter() = TRUE`
    
    # initialize means
    stdv <- 1 / sqrt(self$weight_mu$size(1)) # self$weight_mu$size(1) = out_features
    self$sa_mu <- nn_parameter(torch_normal(1, 1e-2, size = 1))
    self$sb_mu <- nn_parameter(torch_normal(1, 1e-2, size = 1))
    self$atilde_mu <- nn_parameter(torch_normal(1, 1e-2, size = self$atilde_mu$size()))
    self$btilde_mu <- nn_parameter(torch_normal(1, 1e-2, size = self$atilde_mu$size()))
    
    
    # self$z_mu <- nn_parameter(torch_normal(1, 1e-2, size = self$z_mu$size()))      # potential issue (if not considered leaf node anymore?)  wrap in nn_parameter()?
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
    # self$z_logvar <- nn_parameter(torch_normal(mean = log(init_alpha), 1e-2, size = self$in_features)) 
    self$sa_logvar <- nn_parameter(torch_normal(mean = log(.5), 1e-2, size = self$in_features))
    
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
    
    # generate layer activations from Variational specification
    
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
      weight = 
        , 
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
      0.5 * (-self$weight_logvar + self$weight_logvar$exp() + self$weight_mu$pow(2) - 1)
    )
    
    # KL for bias term
    kl_bias <- torch_sum(
      0.5 * (-self$bias_logvar + self$bias_logvar$exp() + self$bias_mu$pow(2) - 1)
    )
    
    # sum
    kl <- kl_z + kl_w_z + kl_bias
    return(kl)
  }
)


