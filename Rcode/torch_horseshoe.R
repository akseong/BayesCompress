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

negKL_lognorm_gamma <- function(mu, logvar, a = 1/2, b = 1){
  # for s_a, alpha_i
  # (LogNormal q || Gamma p)
  # log(b) - 1/b * (exp(1/2 * sig^2 + mu)) + 1/2 * (mu + log(sig^2) + 1 + log(2))
  
  # expr2 <- ((mu$add( logvar$exp()$mul(1/2) ))$exp())$mul(1/b)
  # expr3 <- (mu$add(logvar)$add(1)$add(log(2)))$mul(1/2)
  # return(log(b)$add(-expr2)$add(expr3))
  inner1 <- mu + logvar$exp()$mul(1/2)
  inner2 <- 1 + log(2) + mu + logvar
  log(b) - inner1$exp() / b + inner2 / 2
}

negKL_lognorm_IG <- function(mu, logvar, a = 1/2, b = 1){
  # for s_b, beta_i
  # i.e. (LogNormal q || Inverse-Gamma p)
  # log(b) - 1/b * exp(1/2 * sig^2 - mu) + 1/2 * (- mu + log(sig^2) + 1 + log(2))
  negKL_lognorm_gamma(-mu, logvar, a, b=1)
}


## fcns to compute expectation and variance of lognormals & fcns of lognormals
E_lognorm <- function(mu, logvar){
  # expectation of X ~ logNormal(mu, var)
  # exp(mu + (sig^2)/2)
  # ln_mu <- mu$add( logvar$exp()$mul(0.5) )
  # return(ln_mu$exp())
  # alternatively:
  (mu$add(  logvar$exp()$mul(0.5) ))$exp()
}


V_lognorm <- function(mu, logvar){
  # variance of X ~ logNormal(mu, var)
  # (exp(var) - 1) * exp(2*mu + var)
  expr1 <- logvar$exp()$exp() - 1
  expr2 <- (mu$mul(2))$add(logvar$exp())
  return(expr1$mul(expr2$exp()))
}


E_sqrt_lognorm <- function(mu, logvar){
  # if X ~ LN(mu, var), then X = exp(Y) with Y~N(mu, var)
  # The sqrt(X) = sqrt(exp^Y) = exp^{Y/2} and Y/2 ~ N(mu/2, var/4)
  E_lognorm(mu/2, logvar - log(4))
}

V_sqrt_lognorm <- function(mu, logvar){
  V_lognorm(mu/2, logvar - log(4))
}




## VARIATIONAL DISTRIBUTION PARAMETERS FOR \tilde{z}, s
# z_i = \sqrt(atilde_i btilde_i s_a s_b) = ztilde_i s
## \tilde{z} = \sqrt{\tilde{\alpha}_i \tilde{\beta}_i}    
##   \sim  LogNormal with 
##         mu =     \frac{1}{2}   (\mu_{\tilde{alpha}_i} + \mu_{\tilde{beta}_i} )
##         variance = \frac{1}{4}   (\sigma^2_{\tilde{\alpha}_i} + \sigma^2_{\tilde{\beta}_i}) 
##
## NOTE: for \tilde{z} and s, these are not used in training, but in post-hoc analysis
mu_sqrt_prod_LN <- function(mu_1, mu_2){
  # \frac{mu_1 + mu_2}{2}
  (mu_1$add(mu_2))$mul(1/2)
}

logvar_sqrt_prod_LN <- function(logvar_1, logvar_2){
  # \frac{sig^2_1 + sig^2_2}{4}
  ((logvar_1$exp())$add(logvar_2$exp()))$log() - log(4)
}

log_dropout <- function(hs_layer, type = "local"){
  # calculates dropout rates based on :
  # type == "local":    ztilde = sqrt(atilde btilde)
  # type == "global":    s = sqrt(sa sb)
  # type == "marginal":    z = ztilde * s
  
  if (type == "local"){
    logvar_sum <- hs_layer$atilde_logvar + hs_layer$btilde_logvar
  } else if (type == "global"){
    logvar_sum <- hs_layer$sa_logvar + hs_layer$sb_logvar
  } else if (type == "marginal"){
    logvar_sum <- hs_layer$atilde_logvar + hs_layer$btilde_logvar + hs_layer$sa_logvar + hs_layer$sb_logvar
  }
  
  type_var <- exp(logvar_sum - log(4))
  log_dropout <- log(exp(type_var) - 1)
  return(log_dropout)
}



torch_hs <- nn_module(
  # last modified 1/3/2025
  classname = "horseshoe_layer",
  
  initialize = function(
    in_features, out_features,
    use_cuda = FALSE,
    tau = 1, # scale parameter for global shrinkage prior
    init_weight = NULL,
    init_bias = NULL,
    init_alpha = NULL,
    clip_var = NULL
  ){
    
    self$use_cuda <- use_cuda
    self$tau <- tau
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
    # init_alpha: set atilde_logvar = btilde_logvar = log(2*log(1 + init_alpha))
    # default is init_alpha = 0.5
    if (!is.null(init_alpha)) {
      logvar_abtilde_mu <- log(2*log(1 + init_alpha))
    } else {
      logvar_abtilde_mu <- log(2*log(1.5))
    }
    self$sa_logvar <- nn_parameter(torch_normal(mean = log(.5), 1e-2, size = 1))
    self$sb_logvar <- nn_parameter(torch_normal(mean = log(.5), 1e-2, size = 1))
    self$atilde_logvar <- nn_parameter(torch_normal(mean = logvar_abtilde_mu, 1e-2, size = self$in_features))
    self$btilde_logvar <- nn_parameter(torch_normal(mean = logvar_abtilde_mu, 1e-2, size = self$in_features))
    
    self$ztilde_mu <- mu_sqrt_prod_LN(self$atilde_mu, self$btilde_mu)
    self$ztilde_logvar <- logvar_sqrt_prod_LN(self$atilde_logvar, self$btilde_logvar)
    
    self$s_mu <- mu_sqrt_prod_LN(self$atilde_mu, self$btilde_mu)
    self$s_logvar <- logvar_sqrt_prod_LN(self$sa_mu, self$sb_mu)
    
    self$weight_logvar <- nn_parameter(torch_normal(-9, 1e-2, size = c(self$out_features, self$in_features)))
    self$bias_logvar <- nn_parameter(torch_normal(-9, 1e-2, size = self$out_features))
  },
  
  
  clip_variances = function() {
    if (!is.null(self$clip_var)) {
      self$weight_logvar <- nn_parameter(self$weight_logvar$clamp(max = log(self$clip_var)))
      self$bias_logvar <- nn_parameter(self$bias_logvar$clamp(max = log(self$clip_var)))
    }
  },
  
  
  get_Eztilde_i = function(){
    E_lognorm(
      mu = (self$atilde_mu + self$btilde_mu) / 2, 
      logvar = (self$atilde_logvar + self$btilde_logvar) - log(4)
    )
  },
  
  get_Vztilde_i = function(){
    V_lognorm(
      mu = (self$atilde_mu + self$btilde_mu) / 2, 
      logvar = (self$atilde_logvar + self$btilde_logvar) - log(4)
    )
  },
  
  compute_posterior_param = function() {
    weight_var <- self$weight_logvar$exp()
    # z_var <- self$z_logvar$exp()
    Vz <- V_lognorm(
      mu = (self$atilde_mu + self$btilde_mu + self$sa_mu + self$sb_mu) / 2, 
      logvar = (self$atilde_logvar + self$btilde_logvar + self$sa_logvar + self$sb_logvar) - log(4)
    )
    Ez <- E_lognorm(
      mu = (self$atilde_mu + self$btilde_mu + self$sa_mu + self$sb_mu) / 2, 
      logvar = (self$atilde_logvar + self$btilde_logvar + self$sa_logvar + self$sb_logvar) - log(4)
    )
    
    self$post_weight_var <- Ez$pow(2) * weight_var + Vz * self$weight_mu$pow(2) + Vz * weight_var
    self$post_weight_mu <- self$weight_mu * Ez
    return(list(
      "post_weight_mu" = self$post_weight_mu,
      "post_weight_var" = self$post_weight_var
    ))
  },
  
  get_log_dropout_rates = function(type = "local"){
  # calculates dropout rates based on :
  # type == "local":    ztilde = sqrt(atilde btilde)
  # type == "global":    s = sqrt(sa sb)
  # type == "marginal":    z = ztilde * s
  
  if (type == "local"){
    logvar_sum <- self$atilde_logvar + self$btilde_logvar
  } else if (type == "global"){
    logvar_sum <- self$sa_logvar + self$sb_logvar
  } else if (type == "marginal"){
    logvar_sum <- self$atilde_logvar + self$btilde_logvar + self$sa_logvar + self$sb_logvar
  }
  
  type_var <- exp(logvar_sum - log(4))
  log_dropout <- log(exp(type_var) - 1)
  return(log_dropout)
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
    
    # batch_size <- x$size(1)
    
    # generate layer activations from Variational specification
    log_atilde <- reparameterize(mu = self$atilde_mu, logvar = self$atilde_logvar)
    log_btilde <- reparameterize(mu = self$btilde_mu, logvar = self$btilde_logvar)
    log_sa <- reparameterize(mu = self$sa_mu, logvar = self$sa_logvar)
    log_sb <- reparameterize(mu = self$sb_mu, logvar = self$sb_logvar)
    log_s <- 1/2 * (log_sa + log_sb)
    log_ztilde <- 1/2 * (log_atilde + log_btilde)
    z <- (log_s + log_ztilde)$exp()
    
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

    # KL(q(s_a) || p(s_a));   logNormal || Gamma
    kl_sa <- -negKL_lognorm_gamma(mu = self$sa_mu, logvar = self$sa_logvar, a = 1/2, b = self$tau)
    
    # KL(q(s_b) || p(s_b));   logNormal || invGamma
    kl_sb <- -negKL_lognorm_IG(mu = self$sb_mu, logvar = self$sb_logvar, a = 1/2, b = 1)
    
    # KL(q(atilde) || p(atilde));   logNormal || Gamma
    kl_atilde <- -torch_sum(negKL_lognorm_gamma(mu = self$atilde_mu, logvar = self$atilde_logvar, a = 1/2, b = 1))
    
    # KL(q(btilde) || p(btilde));   logNormal || invGamma
    kl_btilde <- -torch_sum(negKL_lognorm_IG(mu = self$btilde_mu, logvar = self$btilde_logvar, a = 1/2, b = 1))
    
    
    # # KL(q(z) || p(z))
    # kl_z <- -torch_sum(
    #   k1 * nnf_sigmoid(k2 + k3*log_alpha) - 0.5 * nnf_softplus(-log_alpha) - k1
    # )
    
    
    # KL(q(w|z) || p(w|z))
    kl_w_z <- torch_sum(
      0.5 * (-self$weight_logvar + self$weight_logvar$exp() + self$weight_mu$pow(2) - 1)
    )
    
    # KL for bias term
    kl_bias <- torch_sum(
      0.5 * (-self$bias_logvar + self$bias_logvar$exp() + self$bias_mu$pow(2) - 1)
    )
    
    # sum
    kl <- kl_sa + kl_sb + kl_atilde + kl_btilde + kl_w_z + kl_bias
    return(kl)
  }
)


