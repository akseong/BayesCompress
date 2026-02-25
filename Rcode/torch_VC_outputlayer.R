##################################################
## Project:   Varying Coefficient model final layer
## Date:      Feb 25, 2026
## Author:    Arnie Seong
##################################################

# Last VC model layer ----
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
  
  forward = function(vcs, xvars){
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
    
    xzb <- xvars*z*vcs
    return(torch_sum(xzb, dim = 2))
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






















