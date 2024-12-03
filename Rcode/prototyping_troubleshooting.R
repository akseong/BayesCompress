



k1 = 0.63576
k2 = 1.87320
k3 = 1.48695






alph <- 1:1000/100
lalph <- log(alph)


# KL(q(z) || p(z))
klz <- function(x){
  -(k1 * sigmoid(k2 + k3*x) - 0.5 * softplus(-x) - k1)
}


kldiv <- klz(lalph)
plot(kldiv~lalph)
mlnj_net$fc1$compute_posterior_param()
mlnj_net$fc1$post_weight_mu
mlnj_net$fc1$post_weight_var


mlnj_net$fc1$get_log_dropout_rates()
mlnj_net$fc2$get_log_dropout_rates()
summary(as_array(mlnj_net$fc3$get_log_dropout_rates()$exp()))
summary(as_array(mlnj_net$fc2$get_log_dropout_rates()$exp()))
summary(as_array(mlnj_net$fc1$get_log_dropout_rates()$exp()))




# what about centering at the geometric mean?
# geometric mean can be interpreted as 
# the side length of a cube having the same volume
# as a rectangle with these side lengths

la_mat <- log_alpha_mat[, 1:100]
lavec <- la_mat[1, ]
round(exp(lavec - mean(lavec)), 3)

geo_mean <- function(vec){
  lvec <- log(vec)
  exp(lvec - mean(lvec))
}

geo_mean







# stopping methods


# keep track of when:
# - first false positive occurs
# - when get three correct
# - when / if get all 4 correct
# - store tt_mse, MSE, KL
# - store log_alphas

# --- store every 10th?  100th?  model

# - email Karen for paper reference






