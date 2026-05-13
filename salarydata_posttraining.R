##################################################
## Project:   salarydata analysis post training
## Date:      May 12, 2026
## Author:    Arnie Seong
##################################################
library(tidyverse)
library(here)
library(here)
library(torch)
# modified forward portion of torch_horseshoe_klcorrected
source(here("Rcode", "torch_horseshoe_smallbias.R")) 
source(here("Rcode", "sim_functions.R"))
source(here("Rcode", "analysis_fcns.R"))
load(here::here("data", "salary.RData"))

fill_0s <- function(mat, ref_mat = sal_design){
  p <- ncol(ref_mat)
  n_rows <- nrow(mat)
  n_cols <- ncol(mat)
  zero_mat <- matrix(0, nrow = n_rows, ncol = p-n_cols)
  res <- cbind(mat, zero_mat)
  colnames(res) <- colnames(ref_mat)
  return(res)
}


gen_yhats_unsc <- function(nn_mod, Xmat, n_samps = 100, y_mean=y_mean, y_sd=y_sd, qtiles = c(0.025, 0.925), want_yhat_mat = FALSE){
  yhat_mat <- matrix(NA, ncol = n_samps, nrow = nrow(Xmat))
  for (i in 1:n_samps){
    yhat_mat[, i] <- as_array(nn_mod(torch_tensor(Xmat)))
  }  
  if (!is.null(y_mean) & !is.null(y_sd)){
    yhat_mat <- yhat_mat*y_sd + y_mean
  }
  Eyhat <- apply(yhat_mat, 1, mean)
  yhat_qtiles <- apply(yhat_mat, 1, function(X) quantile(X, probs = qtiles))
  
  if (want_yhat_mat){
    return(yhat_mat)
  } else {
    return(cbind(Eyhat, t(yhat_qtiles)))
  }
}


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
    classworker %in% c("Wage/salary", "Government employee", "Self-employed")
  ) %>% 
  mutate(
    occ = factor(occ), 
    logincome = log10(incomewage), 
    race = factor(race), 
    college = ifelse(edu=='CG',1,0)
  ) %>% 
  mutate(
    race = fct_relevel(race, "White"),
    classworker = fct_relevel(classworker, "Wage/salary"),
    occ = fct_relevel(occ, "office")
  ) %>% 
  droplevels() %>% 
  select(logincome, age, female, hispanic, race, college, classworker, difficulty, occ)
numeric_vars <- c("logincome", "age")
numeric_var_columns <- which(colnames(sal_unsc) %in% numeric_vars)

scale_list <- scale_mat(sal_unsc[, numeric_var_columns])
sal <- cbind(scale_list$scaled, sal_unsc[, -numeric_var_columns])

# fname stems
# salary_analysis_h2d4x16
# results/salary_analysis_h2d4x16_nointspec
# salary_analysis_h2d4x16_nointspec_offics

fname_stem <- here::here("final_sims", "results", "salary_analysis_h2d4x16_nointspec_offics")

load(paste0(fname_stem, ".Rdata"))
nn_mod <- torch_load(paste0(fname_stem, ".pt"))

# age normalized in sal_design
x_names <- sim_res$sim_params$x_names
y_mean <- sim_res$sim_params$y_mean
y_sd <- sim_res$sim_params$y_sd
age_mean <- sim_res$sim_params$age_mean
age_sd <- sim_res$sim_params$age_sd
nn_mod_design <- sim_res$sim_params$design

# create matrices to show different trajectories
ages <- 18:65 # length 48
agevec <- (ages-age_mean)/age_sd

# # create expanded datamat to generate predictions
# mo_mat <- expand.grid(
#   "logincome" = 0,
#   "age" = ages, 
#   "female" = c(0,1),
#   "hispanic" = c(0,1),
#   "race" = unique(sal_unsc$race),
#   "college" = unique(sal_unsc$college),
#   "classworker" = unique(sal_unsc$classworker),
#   "difficulty" = c(0,1),
#   "occ" = unique(sal_unsc$occ)
# )
# mo_mat = mo_mat %>% 
#   as_data_frame() %>% 
#   mutate(
#     race = fct_relevel(race, "White"),
#     classworker = fct_relevel(classworker, "Wage/salary"),
#     occ = fct_relevel(occ, "office")
#   )
# mo_mat_design <- model.matrix(as.formula(sim_res$sim_params$formula_str), data = mo_df)[, -1]
# mo_mat_design[,1] <- (mo_mat_design[,1] - age_mean)/age_sd
# colnames(mo_mat_design) == colnames(nn_mod_design)
# 
# 
# n_samps = 100
# yhat_mat <- matrix(NA, ncol = n_samps, nrow = nrow(mo_mat_design))
# for (i in 1:n_samps){
#   yhat_mat[, i] <- as_array(nn_mod(torch_tensor(mo_mat_design)))
# }  
# yhat_mat <- yhat_mat*y_sd + y_mean
# Eyhat <- apply(yhat_mat, 1, mean)
# yhat_qtiles <- apply(yhat_mat, 1, function(X) quantile(X, probs = c(0.025, 0.975)))
# dim(yhat_qtiles)
# mo_mat$Eyhat <- Eyhat
# mo_mat$q.025 <- yhat_qtiles[1, ]
# mo_mat$q.975 <- yhat_qtiles[2, ]
# all_df <- mo_mat[, -1]
# save(all_df, file = paste0(fname_stem, "_dfs.Rdata"))
load(paste0(fname_stem, "_dfs.Rdata"))




# for copy/paste
# all_df %>% 
#   filter(
#     "age"         =       ,   #  18:65   
#     "female"      =       ,   #  0 1       
#     "hispanic"    =       ,   #  0 1         
#     "race"        =       ,   #  Levels: White Asian Black Hawaiian/Pacific Islander Native American     
#     "college"     =       ,   #  0 1        
#     "classworker" =       ,   #  Levels: Wage/salary Government employee Self-employed            
#     "difficulty"  =       ,   #  0, 1           
#     "occ"         =       ,   #      
#   )
# occ levels:
# [1] "office"              "architect/engineer"  "arts/sports/media"   "business operations" "computer/maths"     
# [6] "construction"        "education"           "extraction"          "farming"             "finance"            
# [11] "food"                "health"              "installation"        "legal"               "maintenance"        
# [16] "management"          "personal care"       "production"          "protective"          "sales"              
# [21] "science"             "social service"      "technician"          "transportation" 


MF_race_given_office_wage_nodiff_nocoll_nohisp <- all_df %>% 
  filter(
    # female      ==   0,
    hispanic    ==   0,
    # race        ==   "White",
    college     ==   0,
    classworker ==   "Wage/salary",
    difficulty  ==   0,
    occ         ==   "office"
  )
colnames(all_df)

MF_race_given_office_wage_nodiff_nocoll_nohisp %>% 
  ggplot() +
  geom_line(
    aes(
      y = Eyhat, x = age,
      color = as_factor(female)
    )
  ) + 
  geom_ribbon(
    aes(ymin = q.025, ymax = q.975, 
        x = age,
        # color = as_factor(female)
        fill = as_factor(female)
      ), 
    alpha = 0.2
  ) + 
  facet_wrap(vars(race))
  


table(sal$race)






# default - male, white (vs asian, black, hawaiianPI, native american), no college (vs some college), wage/salary (not gov/self-employed), no reported disability, office
baseline_mat <- fill_0s(matrix(agevec, ncol = 1))


yhat <- nn_mod(torch_tensor(baseline_mat))
yhat_unsc <- yhat*y_sd + y_mean
baseline_plot_df <- data.frame("yhat" = as_array(yhat_unsc), "age" = ages)


gen_yhats_unsc(nn_mod, baseline_mat)

# find data matching covs
colnames(sal_design)
bin_vec <- rep(0, 33)
match_cats <- function(bin_vec, ref_mat = sal_design, match_cols = 2:34){
  match_rows <- which(apply(ref_mat[, match_cols], 1, function(X) all((bin_vec - X)>= 0)))
  return(match_rows)
}


baseline_rows <- match_cats(bin_vec)
baseline_df <- as.data.frame(sal_raw[baseline_rows, ])
baseline_df %>% 
  ggplot() + 
  geom_point(aes(y = logincome, x = age)) + 
  geom_line(data = baseline_plot_df, aes(y = yhat, x = age))










