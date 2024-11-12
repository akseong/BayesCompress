



k1 = 0.63576
k2 = 1.87320
k3 = 1.48695



softplus <- function(x){
  log(1 + exp(x))
}

sigmoid <- function(x){
  1/(1 + exp(-x))
}


alph <- 1:1000/1000
lalph <- log(alph)


# KL(q(z) || p(z))
klz <- function(x){
  -(k1 * sigmoid(k2 + k3*x) - 0.5 * softplus(-x) - k1)
}


y <- klz(lalph)
plot(y~lalph)
mlnj_net$fc1$compute_posterior_param()
mlnj_net$fc1$post_weight_mu
mlnj_net$fc1$post_weight_var


mlnj_net$fc1$get_log_dropout_rates()
mlnj_net$fc2$get_log_dropout_rates()
summary(as_array(mlnj_net$fc3$get_log_dropout_rates()$exp()))
summary(as_array(mlnj_net$fc2$get_log_dropout_rates()$exp()))
summary(as_array(mlnj_net$fc1$get_log_dropout_rates()$exp()))
