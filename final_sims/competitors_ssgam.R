##################################################
## Project:   competitors
## Date:      May 10, 2026
## Author:    Arnie Seong
##################################################

#### setup ----
library(here)
library(tidyr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(BoomSpikeSlab)
library(SoftBart)
library(spikeslab)
library(spikeSlabGAM)

library(torch)
# modified forward portion of torch_horseshoe_klcorrected
source(here("Rcode", "torch_horseshoe_smallbias.R")) 
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))

# retrieve data fcn----
reconstruct_meanfcndat <- function(
    sim_seed,
    sim_params
){
  set.seed(sim_seed)
  torch_manual_seed(sim_seed)
  
  simdat <- sim_meanfcn_data(
    n_obs = sim_params$n_obs, 
    d_in = sim_params$d_in, 
    mut_corr = sim_params$mut_corr,
    genXfcn = sim_params$genXfcn,
    meanfcn = sim_params$meanfcn,
    err_sigma = sim_params$err_sig, 
    round_dig = sim_params$round_dig,
    standardize = FALSE
  )
  
  return(simdat)
}


reconstruct_flistdat <- function(
    sim_seed,
    sim_params
){
  set.seed(sim_seed)
  torch_manual_seed(sim_seed)
  
  simdat <- sim_func_data(
    n_obs = sim_params$n_obs,
    d_in = sim_params$d_in,
    flist = sim_params$flist,
    xlist = NULL,
    err_sigma = sim_params$err_sig,
    use_cuda = sim_params$use_cuda,
    xdist = sim_params$xdist,
    xcov = NULL,
    mut_corr = sim_params$mut_corr,
    standardize = FALSE
  )
  
  return(simdat)
}

fdr_calc <- function(FP, TP) {
  if(FP + TP == 0){return(0)} 
  # precision = 1 - FDR
  res = FP / (FP + TP)
  names(res) = "fdr"
  return(res)
} 
FPR_calc <- function(FP, TN) {
  if(FP + TN == 0){return(0)} 
  res = FP / (FP + TN)
  names(res) = "FPR"
  return(res)
}
TPR_calc <- function(TP, FN) {
  if(TP + FN == 0){return(0)} 
  # sensitivity / recall
  res = TP / (FN + TP)
  names(res) = "TPR"
  return(res)
} 
FNR_calc <- function(FN, TP) {
  if(FN + TP == 0){return(0)} 
  res = FN / (FN + TP)
  names(res) = "FNR"
  return(res)
}
TNR_calc <- function(FP, TN) {
  if(FP + TN == 0){return(0)} 
  # specificity}
  res = TN / (FP + TN)
  names(res) = "TNR"
  return(res)
} 
f1_calc <- function(fdr, tpr){
  prec <- 1-fdr
  if (prec + tpr == 0){return(0)}
  res <- 2 * (prec*tpr) / (prec + tpr)
  names(res) = "f1"
  return(res)
}
metrics_from_decision <- function(est, tru){
  bin_err_num <- binary_err(est, tru) * length(tru)
  FP <- bin_err_num[1]
  TP <- bin_err_num[2]
  FN <- bin_err_num[3]
  TN <- bin_err_num[4]
  
  fdr <- fdr_calc(FP, TP)
  FPR <- FPR_calc(FP, TN) 
  TPR <- TPR_calc(TP, FN) 
  FNR <- FNR_calc(FN, TN) 
  TNR <- TNR_calc(FP, TN)
  f1 <- f1_calc(fdr, TPR)
  c(fdr, FPR, TPR, FNR, TNR, f1 )
}

metrics_err_by_max_bfdr <- function(dropout_vec, true_vec, bfdr_vec){
  # add f1 score:
  # precision = 1-fdr
  # f1 = 2 * (precision*recall)/(precision + recall)
  err_mat <- err_by_max_bfdr(dropout_vec, true_vec, bfdr_vec)$err_mat
  f1 <- apply(err_mat, 1, function(X) f1_calc(fdr=X[2], tpr = X[5]))
  res <- cbind(err_mat, f1)
  colnames(res) <- c("max_bfdr", "fdr", "bfdr", "FPR", "TPR_sens_recall", "FNR", "TN_specificity", "f1")
  res
}





#### COMPILE POSSIBLE DATA SEEDS ----
stem <- here::here("final_sims", "results", "nfdsmallbias_mutcorr0.5_5x165000obs_")
modfcns_TF <- grepl("meanfs", stem)
n_sims = 50
max_bfdr = "arr"
max_bfdrs = c(0.01, 0.05, 0.1, 0.25)
ssgam_cores = 2

if (modfcns_TF){
  reconstruct_fcn <- reconstruct_meanfcndat
  true_vec <- rep(0, 108)
  true_vec[1:8] <- 1
  fname_suffix <- paste0("modfcns_competitors_ssgam", "_bfdr", max_bfdr)
} else {
  reconstruct_fcn <- reconstruct_flistdat
  true_vec <- rep(0, 104)
  true_vec[1:4] <- 1
  fname_suffix <- paste0("origfcns_competitors_ssgam", "_bfdr", max_bfdr)
}

# find seeds
overall_seeds <- as.numeric(c(516, paste0(516, 0:13)))
possible_sim_seeds <- c()
for (i in 1:length(overall_seeds)){
  set.seed(overall_seeds[i])
  possible_sim_seeds <- c(possible_sim_seeds, floor(runif(n = 10, 0, 1000000)))
}

# which of these files exist (some simulations stopped b/c of computer issues)
poss_fnames <- paste0(stem, possible_sim_seeds, ".RData")
exists_TF <- 
  has_corrections_by_layer <- rep(F, length(poss_fnames))

for (i in 1:length(poss_fnames)){
  exists_TF[i] <- file.exists(poss_fnames[i])
}

sum(exists_TF)

sim_seeds <- (possible_sim_seeds[exists_TF])[1:n_sims]

# load one sim to retrieve sim_params
first_sim <- paste0(stem, sim_seeds[1], ".RData")
load(first_sim)
sim_params <- sim_res$sim_params

# construct filename ----
nn_mod <- torch_load(paste0(stem, possible_sim_seeds[exists_TF][1], ".pt"))
hshoe_layers <- grepl("fc", names(nn_mod$children))
det_layers <- grepl("det", names(nn_mod$children))
architecture_str <- paste0("hshoe", sum(hshoe_layers), "det", sum(det_layers))

fname_stem <- paste0(
  architecture_str, "_", 
  sim_res$sim_params$n_obs/1000, "k_",
  n_sims, "sims_", fname_suffix
)

fname <- here::here("final_sims", "compiled", paste0(fname_stem, ".Rdata"))


##### setup storage ----
# PIPs
BHpvals_mat <- pipsmat_ss <- pipsmat_ssgam <- pipsmat_sb <- matrix(NA, nrow = n_sims, ncol = length(true_vec))

# errmats
resarr_colnames <- c("test_mse", "fcn_mse", "time", "max_bfdr", "fdr", "bfdr", "FPR", "TPR_sens_recall", "FNR", "TNR_specificity", "f1")

resarr_ss <- array(NA, dim = c(n_sims, length(resarr_colnames), length(max_bfdrs)))
dimnames(resarr_ss) <- list(paste0("sim_", 1:n_sims), resarr_colnames, paste0("bfdr_",max_bfdrs))

resarr_lm <- resarr_ssgam <- resarr_sb <- resarr_ss

dimnames(resarr_lm)[[2]][4] <- "max_fdr"


for (s_i in 1:n_sims){
  t1_sim <- Sys.time()
  
  simdat <- reconstruct_fcn(sim_seed = sim_seeds[s_i], sim_params)
  simdat_df_raw <- data.frame(
    "y" = as_array(simdat$y),
    "Ey" = as_array(simdat$Ey),
    "x" = as_array(simdat$x)
  )
  n_ttsplit <- sim_params$ttsplit * sim_params$n_obs
  simdat_rawtrain <- simdat_df_raw[1:n_ttsplit, ]
  simdat_rawtest <- simdat_df_raw[1:(sim_params$n_obs - n_ttsplit) + n_ttsplit, ]
  
  # scale train/test split, remove Ey from simdat
  scale_list <- scale_mat(simdat_rawtrain)
  simdat_train <- scale_list$scaled[, -2]
  # Ey_train <- simdat_df_raw$Ey_train
  
  simdat_test <- scale_mat(simdat_rawtest, means = scale_list$means, sds = scale_list$sds)$scaled
  simdat_test <- simdat_test[, -2]
  
  y_train_mean <- mean(simdat_rawtrain$y)
  y_train_sd <- sd(simdat_rawtrain$y)
  # keep Ey_test, y_test unscaled.  unscale the yhats to meet it.
  Ey_test <- simdat_rawtest$Ey
  y_test <- simdat_rawtest$y
  
  
  # # lm ----
  # t1 <- Sys.time()
  # 
  # lm_fit <- lm(y ~ ., data = simdat_train)
  # lm_pvals <- summary(lm_fit)$coef[-1, 4]
  # BH_pvals <- p.adjust(lm_pvals, method = "BH")
  # BH_decisions <- sapply(max_bfdrs, function(X) BH_pvals < X)
  # metrics_lm_mat <- t(apply(BH_decisions, 2, function(X) metrics_from_decision(est = X, tru = true_vec)))
  # colnames(metrics_lm_mat) <- c("fdr", "FPR", "TPR", "FNR", "TNR", "f1")
  # yhat_test <- predict.lm(lm_fit, newdata = simdat_test)
  # yhat_test_unsc <- (yhat_test + y_train_mean)*y_train_sd
  # 
  # mse_test <-  mean((yhat_test_unsc - y_test)^2)
  # fmse_test <- mean((yhat_test_unsc - Ey_test)^2)
  # t2 <- Sys.time()  
  # 
  # ## store
  # BHpvals_mat[s_i, ] <- BH_pvals  # pvals instead of PIPs
  # # "test_mse"   "fcn_mse"    "time"    "max_fdr"   
  # resarr_lm[s_i, 1:3, ] <- c(mse_test, fmse_test, as.numeric(c(t2-t1)))
  # resarr_lm[s_i, 4, ] <- max_bfdrs
  # # "fdr"   "bfdr"  "FPR"    "TPR_sens_recall"    "FNR"    "TN_specificity"    "f1"
  # resarr_lm[s_i, c(5, 7:11), ] <- t(metrics_lm_mat)
  # 
  # 
  # 
  # # Spike-slab-----
  # t1_ss <- Sys.time()
  # modmat <- cbind(1, simdat_train[, -1])
  # prior = IndependentSpikeSlabPrior(modmat, simdat_train$y, 
  #                                   expected.model.size = 20,
  #                                   prior.beta.sd = rep(1, ncol(modmat))) 
  # 
  # ss_fit = lm.spike(y ~ ., data = simdat_train, niter = 1000, prior = prior, ping = 0)
  # ss_summ <- summary(ss_fit)$coef
  # t2_ss <- Sys.time()
  # ## sort spike-slab results to appear in same order as data
  # ss_summ_rnames <- rownames(ss_summ)
  # var_names <- names(simdat_train) # list variables in order appearing in data
  # var_names[1] <- "(Intercept)" # replace "y" with intercept
  # ss_summ_order <- match(var_names, ss_summ_rnames)
  # ss_summ_sorted <- ss_summ[ss_summ_order, ]
  # 
  # # get PIPs, ignore intercept
  # pips_ss <- ss_summ_sorted[-1, 5]
  # metrics_ss <- metrics_err_by_max_bfdr(
  #   dropout_vec = 1-pips_ss, 
  #   true_vec = true_vec, 
  #   bfdr_vec = max_bfdrs
  # )
  # 
  # modmat_test <- cbind(1, simdat_test[, -1])
  # yhat_test <- predict(ss_fit, newdata = modmat_test)
  # yhat_test_unsc <- (yhat_test + y_train_mean)*y_train_sd
  # 
  # mse_test <-  mean((yhat_test_unsc - y_test)^2)
  # fmse_test <- mean((yhat_test_unsc - Ey_test)^2)
  # 
  # # store: 
  # pipsmat_ss[s_i, ] <- pips_ss  
  # resarr_ss[s_i, , ] <- t(
  #   cbind(
  #     mse_test,
  #     fmse_test,
  #     as.numeric(c(t2_ss-t1_ss)),
  #     metrics_ss
  #   )
  # )
  
  
  # SS GAM ----
  f1_string <- paste0("y ~ ", paste0("x.", 1:sim_params$d_in, collapse = " + "))
  f1 <- as.formula(f1_string)
  options(mc.cores = ssgam_cores)

  t1_ssgam <- Sys.time()
  ssgam_fit <- spikeSlabGAM(formula=f1, data=simdat_train)
  yhat_test <- predict(ssgam_fit, newdata = simdat_test)
  t2_ssgam <- Sys.time()

  ssgam_summ <- summary(ssgam_fit)
  posts <- ssgam_summ$trmSummary[-1,1]
  func_posts <- posts[2*1:length(true_vec)]
  lin_posts <- posts[2*1:length(true_vec)-1]
  pips_ssgam <- ifelse(func_posts > lin_posts, func_posts, lin_posts)
  metrics_ssgam <- metrics_err_by_max_bfdr(
    dropout_vec = 1-pips_ssgam,
    true_vec = true_vec,
    bfdr_vec = max_bfdrs
  )

  yhat_test_unsc <- (yhat_test + y_train_mean)*y_train_sd

  mse_test <-  mean((yhat_test_unsc - y_test)^2)
  fmse_test <- mean((yhat_test_unsc - Ey_test)^2)


  # get PIPs, ignore intercept
  pipsmat_ssgam[s_i, ] <- pips_ssgam
  resarr_ssgam[s_i, , ] <- t(cbind(
    mse_test,
    fmse_test,
    as.numeric(c(t2_ssgam-t1_ssgam)),
    metrics_ssgam
  ))

  print(resarr_ssgam[s_i, , ])
  cat("ssgam: "); t2_ssgam - t1_ssgam; cat("\n")
  
  # # softbart ---- 
  # t1_sb <- Sys.time()
  # sbfit <- softbart(
  #   X = simdat_train[, -1],
  #   Y = simdat_train[, 1],
  #   X_test = simdat_test[, -1]
  # )
  # t2_sb <- Sys.time()
  # 
  # yhat_test_unsc <- (sbfit$y_hat_test + y_train_mean)*y_train_sd
  # 
  # mse_test <-  mean((yhat_test_unsc - y_test)^2)
  # fmse_test <- mean((yhat_test_unsc - Ey_test)^2)
  # 
  # # get PIPs, metrics
  # pips_sb <- posterior_probs(sbfit)$post_probs
  # metrics_sb <- metrics_err_by_max_bfdr(
  #   dropout_vec = 1-pips_sb, 
  #   true_vec = true_vec, 
  #   bfdr_vec = max_bfdrs
  # )
  # 
  # # store
  # pipsmat_sb[s_i, ] <- pips_sb
  # resarr_sb[s_i, , ] <- t(
  #   cbind(
  #     mse_test,
  #     fmse_test,
  #     as.numeric(c(t2_sb-t1_sb)),
  #     metrics_sb
  #   )
  # )
  # print(resarr_sb[s_i, , ])
  # cat("softbart: "); t2_sb - t1_sb; cat("\n")
  
  #message ----
  Sys.time()- t1_sim
  cat_color(paste0("simdat ", s_i, " finished; \n", "working on ", fname_stem, "\n"))
}


competitor_list <- list(
  # "BHpvals_mat" = BHpvals_mat,
  # "pipsmat_ss" = pipsmat_ss,
  "pipsmat_ssgam" = pipsmat_ssgam,
  # "pipsmat_sb" = pipsmat_sb,
  # "resarr_lm" = resarr_lm,
  "resarr_ssgam" = resarr_ssgam,
  # "resarr_ss" = resarr_ss,
  # "resarr_sb" = resarr_sb
)

save(competitor_list, file = fname)
cat_color(paste0("results saved to ", fname))







