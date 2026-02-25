##################################################
## Project:   Implementing Horseshoe
## Date:      Dec 9, 2025
## Author:    Arnie Seong
##################################################

# Kaiming initialization implemented



library(torch)

reparameterize <- function(mu, logvar, sampling = TRUE, use_cuda) {
  # for X ~ N(mu, sigma^2), can rewrite as X = mu + sigma * Z
  # "reparam trick" often used to preserve gradient on mu and sigma
  # Last modified 2024/07/16
  # modified 2025/10/27 --- made use_cuda argument extraneous 
  #   - mucked up computation when moving model off cuda / to cuda 
  #     because used stored info to decide whether was cuda or not.
  #   - made extraneous for compatibility with already trained models

  if (sampling) {
    std <- logvar$mul(0.5)$exp_()
    if (mu$is_cuda) {
      eps <- torch_randn(std$size(), device = "cuda", requires_grad = TRUE)
    } else {
      eps <- torch_randn(std$size(), device = "cpu", requires_grad = TRUE)
    }
    return(mu$add(eps$mul(std)))
  } else {
    return(mu)
  }
}



KL_lognorm_gamma <- function(mu, logvar, a = 1/2, b = 1){
  # mu and logvar must be torch_tensors
  # computes KL(logNormal(mu, sig^2)  ||  Gamma(a, b))
  # used for s_a, alpha_i
  # corrected; full KL:
  #   - 1/2 * (log(2) + log(pi) + log(sig^2) + 1)
  #   + log(gamma(a)) + a*(log(b) - mu)
  #   + 1/b * exp(mu + sig^2 / 2)
  # want negative KL
  # expr1 <- -0.5 * (log(2) + log(pi) + logvar + 1)
  # expr2 <- lgamma(a) - (mu$add(-log(b)))$mul(a)
  # expr3 <- (mu$add(  (logvar$exp())$mul(1/2)  ))$exp() / b
  inner1 <- 1 + log(2) + log(pi) + logvar
  inner2 <- mu + logvar$exp()$mul(1/2)
  inner1$mul(-0.5) + lgamma(a) + a*log(b) - mu$mul(a) + inner2$exp() / b
}



KL_lognorm_IG <- function(mu, logvar, a = 1/2, b = 1){
  # for s_b, beta_i
  # i.e. KL(LogNormal q || Inverse-Gamma p)
  # corrected; full KL:
  #   - 1/2 * (log(2) + log(pi) + log(sig^2) + 1)
  #   + log(gamma(a)) + a*(mu - log(b))
  #   + b * exp(-mu + sig^2 / 2)
  # same as KL(LogNormal q || Gamma p) if replace mu with -mu, b with 1/b
  KL_lognorm_gamma(mu = -mu, logvar = logvar, a = a, b = 1/b)
}

# # r versions used for testing
# r_negKL_lognorm_gamma <- function(mu, logvar, a = 1/2, b = 1){
#   # R version - does not work with torch tensors.  Used for testing.
#   # corrected; full KL:
#   #   - 1/2 * (log(2) + log(pi) + log(sig^2) + 1)
#   #   + log(gamma(a)) + a*(log(b) - mu)
#   #   + 1/b * exp(mu + sig^2 / 2)
#   # want negative KL
#   expr1 <- (log(2) + log(pi) + logvar + 1) / 2
#   expr2 <- lgamma(a) + a*(mu - log(b))
#   expr3 <- (mu + exp(logvar)/2) / b
#   
#   return(expr1 - expr2 - expr3)
# }
# r_negKL_lognorm_IG <- function(mu, logvar, a = 1/2, b = 1){
#   # R version for testing
#   r_negKL_lognorm_gamma(-mu, logvar, a, 1/b)
# }
# 
# # test conversion to torch, gradients preserved
# mu <- rnorm(5)
# logvar <- rnorm(5)
# a <- 2
# b <- 2
# 
# r_negKL_lognorm_gamma(mu, logvar, a, b)
# negKL_lognorm_gamma(
#   mu = torch_tensor(mu, requires_grad = TRUE),
#   logvar = torch_tensor(logvar, requires_grad = TRUE),
#   a, b)
# 
# r_negKL_lognorm_IG(mu, logvar, a, b)
# negKL_lognorm_IG(
#   mu = torch_tensor(mu, requires_grad = TRUE),
#   logvar = torch_tensor(logvar, requires_grad = TRUE),
#   a, b)




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

# # fcn testing code
# library(torch)
# self <- list()
# in_features <- 5
# out_features <- 5
# use_cuda <- TRUE
# tau = 1
# clip_var <- NULL
# deterministic = FALSE
# init_weight <- NULL
# init_bias <- NULL


torch_hs_VClast <- nn_module(
  # last modified 10/29/2025 - can now specify initial values for all params
  classname = "VC_horseshoe_final_layer",
  
  initialize = function(
    in_features, out_features,
    use_cuda = FALSE,
    tau_0 = 1, # scale parameter for global shrinkage prior
    init_alpha = NULL,
    init_sa = NULL, 
    init_sb = NULL, 
    init_atilde = NULL, 
    init_btilde = NULL, 
    init_sa_logvar = NULL, 
    init_sb_logvar = NULL, 
    init_atilde_logvar = NULL, 
    init_btilde_logvar = NULL
  ){
    
    self$use_cuda <- use_cuda
    self$tau_0 <- tau_0
    self$in_features <- in_features
    self$out_features <- out_features
    self$devtype <- ifelse(use_cuda, "cuda", "cpu")
    
    #### trainable parameters
    # s = global scale param
    # s^2 = sa*sb
    self$sa_mu <- nn_parameter(torch_randn(1, device = self$devtype))
    self$sa_logvar <- nn_parameter(torch_randn(1, device = self$devtype))
    self$sb_mu <- nn_parameter(torch_randn(1, device = self$devtype))
    self$sb_logvar <- nn_parameter(torch_randn(1, device = self$devtype))
    # z_i_tilde = local scale param
    # z_i_tilde^2 = alpha_tilde * beta_tilde
    self$atilde_mu <- nn_parameter(torch_randn(in_features, device = self$devtype))
    self$atilde_logvar <- nn_parameter(torch_randn(in_features, device = self$devtype))
    self$btilde_mu <- nn_parameter(torch_randn(in_features, device = self$devtype))
    self$btilde_logvar <- nn_parameter(torch_randn(in_features, device = self$devtype))
    
    # composite vars
    # z = \sqrt{  s_a s_b \tilde{\alpha} \tilde{\beta}}
    # s = \sqrt{  s_a s_b  }
    # \tilde{z} = \sqrt{  \tilde{\alpha} \tilde{\beta}  }
    
    # initialize parameters randomly or with pretrained net
    self$reset_parameters(
      init_alpha, init_sa, init_sb, init_atilde, init_btilde, 
      init_sa_logvar, init_sb_logvar, init_atilde_logvar, init_btilde_logvar
    )
    
    # numerical stability param
    self$epsilon <- torch_tensor(1e-8, device = self$devtype)
  },
  
  
  reset_parameters = function(
    init_alpha, init_sa, init_sb, init_atilde, init_btilde, 
    init_sa_logvar, init_sb_logvar, init_atilde_logvar, init_btilde_logvar
  ){
    # specify all for retraining BNN with reduced parameters;
    # specify only init_weight_mu, init_bias_mu for hot start with weights from regular DNN
    #     optionally specify init_alpha to define prior sparsity (default is 1/2)
    
    # initialize means
    stdv <- sqrt(2 / self$in_features) # Kaiming initialization.  changed to scale by # inputs.
    
    if (!is.null(init_sa)) {
      self$sa_mu <- nn_parameter(torch_tensor(init_sa, device = self$devtype))
    } else {
      self$sa_mu <- nn_parameter(torch_normal(log(self$tau_0)/2, 1e-2, size = 1, device = self$devtype))
    }
    
    if (!is.null(init_sb)) {
      self$sb_mu <- nn_parameter(torch_tensor(init_sb, device = self$devtype))
    } else {
      self$sb_mu <- nn_parameter(torch_normal(log(self$tau_0)/2, 1e-2, size = 1, device = self$devtype))
    }
    
    if (!is.null(init_atilde)) {
      self$atilde_mu <- nn_parameter(torch_tensor(init_atilde, device = self$devtype))
    } else {
      self$atilde_mu <- nn_parameter(torch_normal(0, 1e-2, size = self$atilde_mu$size(), device = self$devtype))
    }
    
    if (!is.null(init_btilde)) {
      self$btilde_mu <- nn_parameter(torch_tensor(init_btilde, device = self$devtype))
    } else {
      self$btilde_mu <- nn_parameter(torch_normal(0, 1e-2, size = self$btilde_mu$size(), device = self$devtype))
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
    
    if (!is.null(init_sa_logvar)){
      self$sa_logvar <- nn_parameter(torch_tensor(init_sa_logvar, device = self$devtype))
    } else {
      self$sa_logvar <- nn_parameter(torch_normal(mean = log(.5), 1e-2, size = 1, device = self$devtype))
    }
    
    if (!is.null(init_sb_logvar)){
      self$sb_logvar <- nn_parameter(torch_tensor(init_sb_logvar, device = self$devtype))
    } else {
      self$sb_logvar <- nn_parameter(torch_normal(mean = log(.5), 1e-2, size = 1, device = self$devtype))
    }
    
    if (!is.null(init_atilde_logvar)){
      self$atilde_logvar <- nn_parameter(torch_tensor(init_atilde_logvar, device = self$devtype))
    } else {
      self$atilde_logvar <- nn_parameter(torch_normal(mean = logvar_abtilde_mu, 1e-2, size = self$in_features, device = self$devtype))
    }
    
    if (!is.null(init_btilde_logvar)){
      self$btilde_logvar <- nn_parameter(torch_tensor(init_btilde_logvar, device = self$devtype))
    } else {
      self$btilde_logvar <- nn_parameter(torch_normal(mean = logvar_abtilde_mu, 1e-2, size = self$in_features, device = self$devtype))
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
  
  get_dropout_rates = function(type = "local"){
    # calculates dropout rates based on :
    # type == "local":    ztilde = sqrt(atilde btilde)
    # type == "global":    s = sqrt(sa sb)
    # type == "marginal":    z = ztilde * s
    
    if (type == "local"){
      var_sum <- self$atilde_logvar$exp() + self$btilde_logvar$exp()
    } else if (type == "global"){
      var_sum <- self$sa_logvar$exp() + self$sb_logvar$exp()
    } else if (type == "marginal"){
      var_sum <- self$atilde_logvar$exp() + self$btilde_logvar$exp() + self$sa_logvar$exp() + self$sb_logvar$exp()
    }
    
    type_var <- var_sum / 4
    alpha = type_var$exp() - 1
    return(alpha)
  },
  
  forward = function(betas, xvars){
    # betas generated by penultimate layer
    # want to compute X_i %*% Z %*% betas
    
    # generate layer activations from Variational specification
    log_atilde <- reparameterize(mu = self$atilde_mu, logvar = self$atilde_logvar, use_cuda = self$use_cuda)
    log_btilde <- reparameterize(mu = self$btilde_mu, logvar = self$btilde_logvar, use_cuda = self$use_cuda)
    log_sa <- reparameterize(mu = self$sa_mu, logvar = self$sa_logvar, use_cuda = self$use_cuda)
    log_sb <- reparameterize(mu = self$sb_mu, logvar = self$sb_logvar, use_cuda = self$use_cuda)
    log_s <- 1/2 * (log_sa + log_sb)
    log_ztilde <- 1/2 * (log_atilde + log_btilde)
    z <- (log_s + log_ztilde)$exp()
    # log_z <- reparameterize(
    #   mu = 1/2 * (self$atilde_mu + self$btilde_mu + self$sa_mu + self$sb_mu),
    #   logvar = (self$atilde_logvar$exp() + self$btilde_logvar$exp() + self$sa_logvar$exp() + self$sb_logvar$exp())$log() - log(4),
    #   use_cuda = self$use_cuda
    # )
    # z <- log_z$exp()
    
    xz <- xvars*z

    return(
      nnf_linear(
        input = xz, 
        weight = betas, 
        bias = 0
      )
    )
  },
  
  
  get_kl = function() {
    
    # KL(q(s_a) || p(s_a));   logNormal || Gamma
    kl_sa <- KL_lognorm_gamma(mu = self$sa_mu, logvar = self$sa_logvar, a = 1/2, b = self$tau_0)
    
    # KL(q(s_b) || p(s_b));   logNormal || invGamma
    kl_sb <- KL_lognorm_IG(mu = self$sb_mu, logvar = self$sb_logvar, a = 1/2, b = 1)
    
    # KL(q(atilde) || p(atilde));   logNormal || Gamma
    kl_atilde <- torch_sum(KL_lognorm_gamma(mu = self$atilde_mu, logvar = self$atilde_logvar, a = 1/2, b = 1))
    
    # KL(q(btilde) || p(btilde));   logNormal || invGamma
    kl_btilde <- torch_sum(KL_lognorm_IG(mu = self$btilde_mu, logvar = self$btilde_logvar, a = 1/2, b = 1))
    
    # sum
    kl <- kl_sa + kl_sb + kl_atilde + kl_btilde
    return(kl)
  }
)






















