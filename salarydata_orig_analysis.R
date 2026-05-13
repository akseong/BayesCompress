


#These data were obtained from the Current Population Survey at https://www.census.gov/programs-surveys/cps.html
#
#To use these data you should add a reference to:
#
#Flood, Sarah and King, Miriam and Rodgers, Renae and Ruggles, Steven and Warren, J Robert (2020).
#Integrated Public Use Microdata Series, Current Population Survey: Version 7.0 [dataset]
#Minneapolis, MN: IPUMS, 10.18128/D030.V7.0
#
# See https://cps.ipums.org/cps/citation.shtml for the rules regarding citation and use of the CPS database

library(tidyverse)
library(here)
# source("https://www.ics.uci.edu/~akseong/Rcode/EDA_R_functions.R")

load(here::here("data", "salary.RData"))

data= filter(salary, incomewage>=500, age>=18, age<=65, employment=="At work", !(occ %in% c("unemployed/neverworked","military")), hoursworked>=35, hoursworked<=40, wkstat=="Full-time") |>
  mutate(occ= factor(occ), logincome= log10(incomewage), black= ifelse(race=='Black',1,0), college= ifelse(edu=='CG',1,0), classworker=factor(classworker)) |>
  mutate(female_black= female * black) |>
  select(c(logincome, age, female, hispanic, black, college, occ, classworker, female_black))

table(salary$race)
# pre-process dataset ----
sal_unsc <- salary %>% 
  filter(
    incomewage>=500,
    age >= 18,
    age <= 65,
    employment == "At work",
    !(occ %in% c("unemployed/never worked","military")),
    hoursworked >= 35,
    hoursworked <= 40,
    wkstat == "Full-time",
    classworker %in% c("Wage/salary", "Government employee", "Self-employed"),
    race %in% c("Asian", "Black", "White")
  ) %>% 
  mutate(
    occ = factor(occ), 
    logincome = log10(incomewage), 
    black = ifelse(race=='Black',1,0),
    college = ifelse(edu=='CG',1,0)
  ) %>% 
  mutate(
    race = fct_relevel(race, "White"),
    classworker = fct_relevel(classworker, "Wage/salary"),
    occ = fct_relevel(occ, "office")
  ) %>% 
  select(logincome, age, female, hispanic, black, college, occ, classworker) %>% 
  droplevels()


# check
names(sal_unsc)
n_obs <- nrow(sal_unsc)
n_obs
str(sal_unsc)
table(sal_unsc$female)
table(sal_unsc$occ)
table(sal_unsc$classworker)
table(sal_unsc$black)

library(here)
library(torch)
# modified forward portion of torch_horseshoe_klcorrected
source(here("Rcode", "torch_horseshoe_smallbias.R")) 
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))
# normalize data
numeric_vars <- c("logincome", "age")
numeric_var_columns <- which(colnames(sal_unsc) %in% numeric_vars)

scale_list <- scale_mat(sal_unsc[, numeric_var_columns])
sal <- cbind(scale_list$scaled, sal_unsc[, -numeric_var_columns])

str(sal)
# logincome ~ age + age:female + age:hispanic + age:race + age:college + age:classworker + age:difficulty + age:occ

# need to include intercept to get reference-coded design matrix 
# formula_str <- "logincome ~ 1 + age + age:female + age:hispanic + age:race + age:college + age:classworker + age:difficulty + age:occ"
# formula_maineffects_str <- "logincome ~ 1 + age + age*female + age*hispanic + age*race + age*college + age*classworker + age*difficulty + age*occ"
formula_str <- "logincome ~ 1 + age + female + hispanic + black + college + classworker + occ"
sal_design <- model.matrix(as.formula(formula_str), data = sal)
colnames(sal_design)

# remove intercept
sal_design <- sal_design[, -1]
colnames(sal_design)

# test-train split
ttsplit <- 9/10
n_obs <- nrow(sal_design)
x_names <- colnames(sal_design)
set.seed(516)
test_inds <- sample(1:n_obs, size = floor(n_obs*(1-ttsplit)))
train_inds <- setdiff(1:n_obs, test_inds)
n_train <- length(train_inds)
n_test <- length(test_inds)

# INITIALIZE NETWORK ----
if (torch::cuda_is_available()){
  use_cuda <- TRUE
  message("Default tensor device set to GPU (CUDA).")
} else {
  use_cuda <- FALSE
  message("Default tensor device remains CPU.")
}


prior_tau <- tau0_PV(
  p_0 = ncol(sal_design)/2, 
  d = ncol(sal_design),
  sig = 1, 
  n = n_train)

agnostic_tau <- tau0_PV(
  p_0 = 1, d = 2, sig = 1, n = n_train
)
  
save_stem <- here::here("final_sims", "results", "salary_analysis_h2d4x16_orig")
sim_params <- list(
  # train_params
  seed = 516,
  train_epochs = 5e4,
  report_every = 1e3,
  kl_scheduler = kl_weight_cosine,
  kl_warmup_frac = 0.4,
  n_mc_samples = 5,
  prior_tau = prior_tau,
  agnostic_tau = agnostic_tau,
  use_cuda = use_cuda,
  save_stem = save_stem,
  
  # data_params
  n_obs = n_obs,
  sal_unsc = sal_unsc,
  sal = sal,
  design = sal_design,
  formula_str = formula_str,
  ttsplit = ttsplit,
  x_names = x_names,
  y_mean = scale_list$means[1],
  y_sd = scale_list$sds[1],
  age_mean = scale_list$means[2],
  age_sd = scale_list$sds[2],
  
  #architecture
  d_in = ncol(sal_design),
  d_1 = 16,
  d_2 = 16,
  d_3 = 16,
  d_4 = 16,
  d_5 = 16,
  d_out = 1
)

# tensor ttsplit ----
x_train <- torch_tensor(sal_design[train_inds, ], device = dev_select(use_cuda))
x_test <- torch_tensor(sal_design[test_inds, ], device = dev_select(use_cuda))
y_train <- torch_tensor(sal$logincome[train_inds], device = dev_select(use_cuda))$unsqueeze(dim=2)
y_test <- torch_tensor(sal$logincome[test_inds], device = dev_select(use_cuda))$unsqueeze(dim=2)


## define model
MLHS <- nn_module(
  "MLHS",
  initialize = function() {
    self$fc1 = torch_hs(    
      in_features = sim_params$d_in, 
      out_features = sim_params$d_1,
      use_cuda = sim_params$use_cuda,
      tau_0 = sim_params$prior_tau,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    self$fc2 = torch_hs(
      in_features = sim_params$d_1,
      out_features = sim_params$d_2,
      use_cuda = sim_params$use_cuda,
      tau_0 = agnostic_tau,
      init_weight = NULL,
      init_bias = NULL,
      init_alpha = 0.9,
      clip_var = TRUE
    )
    
    # self$fc3 = torch_hs(
    #   in_features = sim_params$d_2,
    #   out_features = sim_params$d_out,
    #   use_cuda = sim_params$use_cuda,
    #   tau_0 = agnostic_tau,
    #   init_weight = NULL,
    #   init_bias = NULL,
    #   init_alpha = 0.9,
    #   clip_var = TRUE
    # )
    
    self$det1 = nn_linear(
      sim_params$d_2,
      sim_params$d_3
    )
    
    self$det2 = nn_linear(
      sim_params$d_3,
      sim_params$d_4
    )
    
    self$det3 = nn_linear(
      sim_params$d_4,
      sim_params$d_5
    )
    
    self$det4 = nn_linear(
      sim_params$d_5,
      sim_params$d_out
    )
    
    if (sim_params$use_cuda){
      self$det1$cuda()
      self$det2$cuda()
      self$det3$cuda()
      self$det4$cuda()
    }
  },
  
  forward = function(x) {
    x %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      nnf_relu() %>%
      # self$fc3()
      self$det1() %>%
      nnf_relu() %>%
      self$det2() %>%
      nnf_relu() %>%
      self$det3() %>%
      nnf_relu() %>%
      self$det4()
  },
  
  get_model_kld = function(){
    kl1 = self$fc1$get_kl()
    kl2 = self$fc2$get_kl()
    # kl3 = self$fc3$get_kl()
    # kl4 = self$fc4$get_kl()
    # kl5 = self$fc5$get_kl()
    kld = kl1 + kl2 #+ kl3 # + kl4 + kl5
    return(kld)
  }
)


# setup storage ----
report_epochs <- seq(
  sim_params$report_every, 
  sim_params$train_epochs, 
  by = sim_params$report_every
)

loss_mat <- matrix(
  NA, 
  nrow = length(report_epochs),
  ncol = 5
)
colnames(loss_mat) <- c("kl", "mse_train", "mse_test", "kl_raw", "kl_weight")
rownames(loss_mat) <- report_epochs

# store: alphas, kappas
alpha_mat <- matrix(
  NA, 
  nrow = length(report_epochs),
  ncol = sim_params$d_in
)
colnames(alpha_mat) <- colnames(sal_design)
rownames(alpha_mat) <- report_epochs
kappa_local_mat <- 
  kappa_mat <- 
  kappa_tc_mat <- 
  kappa_sn_mat <-
  kappa_sntc_mat <- 
  kappa_fc_mat <- alpha_mat


## TRAINING LOOP ----

#initialize model and optim
torch_manual_seed(sim_params$seed)
model_fit <- MLHS()
optim_model_fit <- optim_adam(model_fit$parameters)

# kl annealing
kl_warmup_epochs <- round(sim_params$train_epochs * sim_params$kl_warmup_frac)

## initialize training params
epoch <- 1
loss <- torch_tensor(1, device = dev_select(sim_params$use_cuda))

## stop criteria
stop_epochs <- c()
## test_mse_storage
mse_test <- torch_tensor(0, device = dev_select(sim_params$use_cuda))
loss_test <- torch_tensor(1, device = dev_select(sim_params$use_cuda)) 

n_mc <- ifelse(
  !is.null(sim_params$n_mc_samples),
  sim_params$n_mc_samples, 1
)


while(epoch <= sim_params$train_epochs){

  # zero out previous gradients
  optim_model_fit$zero_grad()
  
  # accumulate mse gradients over s MC samples
  mse_accum <- torch_tensor(0, device = dev_select(sim_params$use_cuda))
  for (s in 1:n_mc){
    # get mse using n_mc MC samples
    yhat_train <- model_fit(x_train)
    mse_s <- nnf_mse_loss(yhat_train, y_train) / n_mc
    mse_s$backward()
  }
  
  # fit & metrics
  kl_raw <- model_fit$get_model_kld() / n_train
  kl_weight <- ifelse(
    !is.null(sim_params$kl_scheduler),
    sim_params$kl_scheduler(epoch, kl_warmup_epochs),
    1
  )
  kl <- kl_weight * kl_raw
  kl$backward()
  
  # update weights
  optim_model_fit$step()
  mse <- mse_s * n_mc # approximate, based only on last sample's mse
  loss <- mse + kl

  time_to_report <- epoch!=0 & (epoch %% sim_params$report_every == 0)
  if (time_to_report){
    row_ind <- epoch %/% sim_params$report_every
    
    # compute test loss 
    model_fit$eval()                    # switches ALL layers to deterministic
    with_no_grad({
      yhat_test <- model_fit(x_test)
      mse_test <- nnf_mse_loss(yhat_test, y_test)
    })
    model_fit$train()
    
    loss_mat[row_ind, ] <- c(kl$item(), mse$item(), mse_test$item(), kl_raw$item(), kl_weight)
    dropout_alphas <- model_fit$fc1$get_dropout_rates()
    alpha_mat[row_ind, ] <- as_array(dropout_alphas)
    kappas <- get_kappas(model_fit$fc1)
    kappa_mat[row_ind, ] <- kappas
    
    kappas_local <- get_kappas(model_fit$fc1, type = "local")
    kappa_local_mat[row_ind, ] <- kappas_local
    
    # corrected param kappas
    kappas_tc <- get_kappas_taucorrected(model_fit)
    kappas_fc <- get_kappas_frobcorrected(model_fit)
    kappas_sn <- get_kappas_compositespecnorm(model_fit)
    kappas_sntc <- get_kappas_sntau(model_fit)
    kappa_sn_mat[row_ind, ] <- kappas_sn
    kappa_tc_mat[row_ind, ] <- kappas_tc
    kappa_fc_mat[row_ind, ] <- kappas_fc
    kappa_sntc_mat[row_ind, ] <- kappas_sntc
  } # end result storing and updating
  
  # report
  if (!time_to_report & (epoch %% sim_params$report_every == 1)){cat("Training till next report:")}
  if (!time_to_report & (epoch %% round(sim_params$report_every/100) == 1)){cat("#")}
  if (time_to_report){
    cat(
      "\n Epoch:", epoch,
      "MSE + KL/n =", round(mse$item(), 5), "+", round(kl$item(), 5),
      "=", round(loss$item(), 4),
      " (kl_weight:", round(kl_weight, 2), ")",
      "\n",
      "train mse:", round(mse$item(), 4),
      "; test_mse:", round(mse_test$item(), 4),
      sep = " "
    )
    
    # report global shrinkage params (s^2 or tau^2)
    s_sq1 <- get_s_sq(model_fit$fc1)
    s_sq2 <- get_s_sq(model_fit$fc2)
    
    cat(
      "\n s_sq1 = ", round(s_sq1, 5),
      "; s_sq2 = ", round(s_sq2, 5),
      sep = ""
    )
    cat("\n alphas: ", round(as_array(dropout_alphas), 2), "\n")
    display_alphas <- ifelse(
      as_array(dropout_alphas) <= sim_params$alpha_thresh,
      round(as_array(dropout_alphas), 3),
      "."
    )

    cat("\n local kappas: ", round(kappas_local, 2), "\n")
    cat("\n global kappas: ", round(kappas, 2), "\n")
    cat("\n tau-corrected kappas: ", round(kappas_tc, 2), "\n")
    cat("\n specnorm(composite) kappas: ", round(kappas_sn, 2), "\n")
    cat("\n fc2&specnorm(composite) kappas: ", round(kappas_sntc, 2), "\n")
    cat(" \n \n")
  }
  
  epoch <- epoch + 1
}

### compile results ----
sim_res <- list(
  "loss_mat" = loss_mat,
  "alpha_mat" = alpha_mat,
  "kappa_mat" = kappa_mat,
  "kappa_tc_mat" = kappa_tc_mat,
  "kappa_fc_mat" = kappa_fc_mat,
  "kappa_sn_mat" = kappa_sn_mat,
  "kappa_sntc_mat" = kappa_sntc_mat,
  "kappa_local_mat" = kappa_local_mat
)
# save model
save_mod_path <- paste0(save_stem, ".pt")
save_res_path <- paste0(save_stem, ".Rdata")
torch_save(model_fit, path = save_mod_path)
cat_color(txt = paste0("model saved: ", save_mod_path))
sim_res$mod_path = save_mod_path

# save results
sim_res$sim_params <- sim_params
save(sim_res, file = save_res_path)
cat_color(txt = paste0("sim results saved: ", save_res_path))


