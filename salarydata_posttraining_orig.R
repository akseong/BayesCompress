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

fname_stem <- here::here("final_sims", "results", "salary_analysis_h2d4x16_orig")

load(paste0(fname_stem, ".Rdata"))
nn_mod <- torch_load(paste0(fname_stem, ".pt"))


# age normalized in sal_design
sal_unsc <- sim_res$sim_params$sal_unsc
sal <- sim_res$sim_params$sal
sal_design <- sim_res$sim_params$design

names(sal_design)

x_names <- sim_res$sim_params$x_names
y_mean <- sim_res$sim_params$y_mean
y_sd <- sim_res$sim_params$y_sd
age_mean <- sim_res$sim_params$age_mean
age_sd <- sim_res$sim_params$age_sd

# create matrices to show different trajectories
ages <- 18:65 # length 48
agevec <- (ages-age_mean)/age_sd

# create expanded datamat to generate predictions
all_df <- expand.grid(
  "logincome" = 0,
  "age" = ages,
  "female" = c(0,1),
  "hispanic" = c(0,1),
  "black" = c(0,1),
  "college" = c(0,1),
  "classworker" = unique(sal_unsc$classworker),
  "occ" = unique(sal_unsc$occ)
) %>%
  mutate(
    classworker = fct_relevel(classworker, "Wage/salary"),
    occ = fct_relevel(occ, "office")
  )
all_design <- model.matrix(as.formula(sim_res$sim_params$formula_str), data = all_df)[, -1]
all_design[,1] <- (all_design[,1] - age_mean)/age_sd
colnames(all_design) == colnames(sim_res$sim_params$design)


n_samps = 100
yhat_mat <- matrix(NA, ncol = n_samps, nrow = nrow(all_design))
for (i in 1:n_samps){
  yhat_mat[, i] <- as_array(nn_mod(torch_tensor(all_design)))
}
yhat_mat <- yhat_mat*y_sd + y_mean
Eyhat <- apply(yhat_mat, 1, mean)
yhat_qtiles <- apply(yhat_mat, 1, function(X) quantile(X, probs = c(0.025, 0.975)))
dim(yhat_qtiles)
all_df$Eyhat <- Eyhat
all_df$q.025 <- yhat_qtiles[1, ]
all_df$q.975 <- yhat_qtiles[2, ]
all_df <- all_df[, -1]
save(all_df, file = paste0(fname_stem, "_dfs.Rdata"))
load(paste0(fname_stem, "_dfs.Rdata"))




# for copy/paste
# all_df %>% 
#   filter(
#     female      = 0                 ,   #  0 1       
#     hispanic    = 0                 ,   #  0 1         
#     black       = 0                 ,   #  0 1
#     college     = 0                 ,   #  0 1        
#     classworker = "Wage/salary"     ,   #  Levels: Wage/salary Government employee Self-employed            
#     occ         = "office"          ,   #      
#   )
# occ levels:
# [1] "office"              "architect/engineer"  "arts/sports/media"   "business operations" "computer/maths"     
# [6] "construction"        "education"           "extraction"          "farming"             "finance"            
# [11] "food"                "health"              "installation"        "legal"               "maintenance"        
# [16] "management"          "personal care"       "production"          "protective"          "sales"              
# [21] "science"             "social service"      "technician"          "transportation" 


fhb <- all_df %>% 
  filter(
    # female      ==   0,
    # hispanic    ==   0,
    # black       ==   0,
    college     ==   0,
    classworker ==   "Wage/salary",
    occ         ==   "office"
  ) %>% 
  mutate(
    hb = case_when(
      hispanic == 0 & black == 0 ~ "non-Hispanic, non-Black",
      hispanic == 0 & black == 1 ~ "non-Hispanic, Black",
      hispanic == 1 & black == 1 ~ "Hispanic, Black",
      hispanic == 1 & black == 0 ~ "Hispanic, non-Black"
    )
  )


fhb %>% 
  ggplot() +
  geom_line(
    aes(
      y = Eyhat, x = age,
      color = as_factor(female),
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
  facet_wrap(vars(hb))
  


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










