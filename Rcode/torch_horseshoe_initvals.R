torch_hs <- nn_module(
  # last modified 10/10/2025
  classname = "horseshoe_layer",
  
  initialize = function(
    in_features, out_features,
    use_cuda = FALSE,
    tau = 1, # scale parameter for global shrinkage prior
    init_sa_mu = NULL,
    init_sb_mu = NULL,
    init_atilde_mu = NULL,
    init_btilde_mu = NULL,
    init_weight = NULL,
    init_bias = NULL,
    init_alpha = NULL,
    clip_var = NULL, 
    deterministic = FALSE
  ){
    
    self$use_cuda <- use_cuda
    self$tau <- tau
    self$in_features <- in_features
    self$out_features <- out_features
    self$clip_var <- clip_var
    self$deterministic <- deterministic
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
    
    # weight dist'n params
    self$weight_mu <- nn_parameter(torch_randn(out_features, in_features, device = self$devtype))
    self$weight_logvar <- nn_parameter(torch_randn(out_features, in_features, device = self$devtype))
    self$bias_mu <- nn_parameter(torch_randn(out_features, device = self$devtype))
    self$bias_logvar <- nn_parameter(torch_randn(out_features, device = self$devtype))
    
    
    # composite vars
    # z = \sqrt{  s_a s_b \tilde{\alpha} \tilde{\beta}}
    # s = \sqrt{  s_a s_b  }
    # \tilde{z} = \sqrt{  \tilde{\alpha} \tilde{\beta}  }
    
    # initialize parameters randomly or with pretrained net
    self$reset_parameters(init_weight, init_bias, init_alpha)
    
    # numerical stability param
    self$epsilon <- torch_tensor(1e-8, device = self$devtype)
  },
  
  
  reset_parameters = function(init_sa_mu, init_sb_mu, init_atilde_mu, init_btilde_mu, init_weight, init_bias, init_alpha){
    # specify all for retraining BNN with reduced parameters;
    # specify only init_weight_mu and init_bias_mu for hot start with weights from regular DNN
    
    # initialize means
    stdv <- 1 / sqrt(self$weight_mu$size(1)) # self$weight_mu$size(1) = out_features
    # self$sa_mu <- nn_parameter(torch_normal(1, 1e-2, size = 1, device = self$devtype))
    # self$sb_mu <- nn_parameter(torch_normal(1, 1e-2, size = 1, device = self$devtype))
    # self$atilde_mu <- nn_parameter(torch_normal(1, 1e-2, size = self$atilde_mu$size(), device = self$devtype))
    # self$btilde_mu <- nn_parameter(torch_normal(1, 1e-2, size = self$atilde_mu$size(), device = self$devtype))
    
    
    if (!is.null(init_sa_mu)) {
      self$sa_mu <- nn_parameter(torch_tensor(init_sa_mu, device = self$devtype))
    } else {
      self$sa_mu <- nn_parameter(torch_normal(1, 1e-2, size = 1, device = self$devtype))
    }
    
    if (!is.null(init_sb_mu)) {
      self$sb_mu <- nn_parameter(torch_tensor(init_sb_mu, device = self$devtype))
    } else {
      self$sb_mu <- nn_parameter(torch_normal(1, 1e-2, size = 1, device = self$devtype))
    }
    
    if (!is.null(init_atilde_mu)) {
      self$atilde_mu <- nn_parameter(torch_tensor(init_atilde_mu, device = self$devtype))
    } else {
      self$atilde_mu <- nn_parameter(torch_normal(1, 1e-2, size = self$atilde_mu$size(), device = self$devtype))
    }
    
    if (!is.null(init_btilde_mu)) {
      self$btilde_mu <- nn_parameter(torch_tensor(init_btilde_mu, device = self$devtype))
    } else {
      self$btilde_mu <- nn_parameter(torch_normal(1, 1e-2, size = self$btilde_mu$size(), device = self$devtype))
    }
    
    # self$z_mu <- nn_parameter(torch_normal(1, 1e-2, size = self$z_mu$size()))      # potential issue (if not considered leaf node anymore?)  wrap in nn_parameter()?
    if (!is.null(init_weight)) {
      self$weight_mu <- nn_parameter(torch_tensor(init_weight, device = self$devtype))
    } else {
      self$weight_mu <- nn_parameter(torch_normal(0, stdv, size = self$weight_mu$size(), device = self$devtype))
    }
    
    if (!is.null(init_bias)) {
      self$bias_mu <- nn_parameter(torch_tensor(init_bias, device = self$devtype))
    } else {
      self$bias_mu <- nn_parameter(torch_zeros(self$out_features, device = self$devtype))
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
    
    if (!is.null(init_atilde_logvar)){
      self$atilde_logvar <- nn_parameter(torch_tensor(init_atilde_logvar, device = self$devtype))
    } else {
      self$atilde_logvar <- nn_parameter(torch_normal(mean = logvar_abtilde_mu, 1e-2, size = self$in_features, device = self$devtype))
    }
    
    if (!is.null(init_atilde_logvar)){
      self$btilde_logvar <- nn_parameter(torch_tensor(init_btilde_logvar, device = self$devtype))
    } else {
      self$btilde_logvar <- nn_parameter(torch_normal(mean = logvar_abtilde_mu, 1e-2, size = self$in_features, device = self$devtype))
    }
  
    
    
    
    self$sa_logvar <- nn_parameter(torch_normal(mean = log(.5), 1e-2, size = 1, device = self$devtype))
    self$sb_logvar <- nn_parameter(torch_normal(mean = log(.5), 1e-2, size = 1, device = self$devtype))
    
    
    self$weight_logvar <- nn_parameter(torch_normal(-9, 1e-2, size = c(self$out_features, self$in_features), device = self$devtype))
    self$bias_logvar <- nn_parameter(torch_normal(-9, 1e-2, size = self$out_features, device = self$devtype))
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
  
  forward = function(x){
    if (self$deterministic) {
      cat("argument deterministic = TRUE.  Should not be used for training")
      return(
        nnf_linear(
          input = x, 
          weight = self$weight_mu, 
          bias = self$bias_mu
        )
      )
    }
    
    # generate layer activations from Variational specification
    log_atilde <- reparameterize(mu = self$atilde_mu, logvar = self$atilde_logvar, use_cuda = self$use_cuda)
    log_btilde <- reparameterize(mu = self$btilde_mu, logvar = self$btilde_logvar, use_cuda = self$use_cuda)
    log_sa <- reparameterize(mu = self$sa_mu, logvar = self$sa_logvar, use_cuda = self$use_cuda)
    log_sb <- reparameterize(mu = self$sb_mu, logvar = self$sb_logvar, use_cuda = self$use_cuda)
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