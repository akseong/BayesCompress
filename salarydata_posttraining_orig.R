##################################################
## Project:   salarydata analysis post training
## Date:      May 12, 2026
## Author:    Arnie Seong
##################################################
library(tidyverse)
library(here)
library(here)
library(torch)
library(latex2exp)
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
    # race = fct_relevel(race, "White"),
    classworker = fct_relevel(classworker, "Wage/salary")
    # occ = fct_relevel(occ, "office")
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
save(all_df, file = paste0(fname_stem, "_all_df.Rdata"))
load(paste0(fname_stem, "_all_df.Rdata"))

all_df <- all_df %>% 
  mutate(
    gender = ifelse(female == 1, "Female", "Male"),
    hb = fct_relevel(
      case_when(
        hispanic == 0 & black == 0 ~ "non-Hispanic, non-Black",
        hispanic == 0 & black == 1 ~ "non-Hispanic, Black",
        hispanic == 1 & black == 1 ~ "Hispanic, Black",
        hispanic == 1 & black == 0 ~ "Hispanic, non-Black"
      ), c("Hispanic, Black", "non-Hispanic, non-Black", "Hispanic, non-Black", "non-Hispanic, Black")
    )
  )



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



## salaried office workers, no college degree ----
fhb <- all_df %>% 
  filter(
    # female      ==   0,
    # hispanic    ==   0,
    # black       ==   0,
    college     ==   0,
    classworker ==   "Wage/salary",
    occ         ==   "office"
  ) 


fhb_plt <- fhb %>% 
  ggplot() +
  geom_line(
    aes(
      y = Eyhat, x = age,
      color = gender,
    )
  ) + 
  geom_ribbon(
    aes(ymin = q.025, ymax = q.975, 
        x = age,
        # color = as_factor(female)
        fill = gender
      ), 
    alpha = 0.2
  ) + 
  facet_wrap(vars(hb)) + 
  labs(
    # title = TeX("Mean $log_{10}$ income trajectories"),
    subtitle = "Salaried office workers, no college degree",
    y = TeX("Estimated mean $log_{10}$ income"),
    x = "Age (years)",
    color = "",
    fill = ""
  ) + 
  ylim(c(4, 4.87))+ 
  theme(
    legend.position = c(0.9, 0.1)
  )
fhb_plt
# 10^4.2 ~ 16k, 10^4.4 ~ 25k, 10^4.6 ~ 40k
ggsave(plot = fhb_plt, filename = "fhb_plot_notitle.png" )


## salaried office workers; college degree ----
fhb_coll <- all_df %>% 
  filter(
    # female      ==   0,
    # hispanic    ==   0,
    # black       ==   0,
    college     ==   1,
    classworker ==   "Wage/salary",
    occ         ==   "office"
  )


fhb_coll_plt <- fhb_coll %>% 
  ggplot() +
  geom_line(
    aes(
      y = Eyhat, x = age,
      color = gender,
    )
  ) + 
  geom_ribbon(
    aes(ymin = q.025, ymax = q.975, 
        x = age,
        # color = as_factor(female)
        fill = gender
    ), 
    alpha = 0.2
  ) + 
  ylim(c(4, 4.87))+ 
  facet_wrap(vars(hb)) + 
  labs(
    # title = TeX("Mean $log_{10}$ income trajectories"),
    subtitle = "Salaried office workers with college degree",
    y = TeX("Estimated mean $log_{10}$ income"),
    x = "Age (years)",
    color = "",
    fill = ""
  ) + 
  theme(
    legend.position = c(0.9, 0.1)
  )
fhb_coll_plt

ggsave(plot = fhb_coll_plt, filename = "fhb_coll_plot_notitle.png" )


## salaried sales workers, no college degree ----
fhb_sales <- all_df %>% 
  filter(
    # female      ==   0,
    # hispanic    ==   0,
    # black       ==   0,
    college     ==   0,
    classworker ==   "Wage/salary",
    occ         ==   "sales"
  )

fhb_sales_plt <- fhb_sales %>% 
  ggplot() +
  geom_line(
    aes(
      y = Eyhat, x = age,
      color = gender,
    )
  ) + 
  geom_ribbon(
    aes(ymin = q.025, ymax = q.975, 
        x = age,
        # color = as_factor(female)
        fill = gender
    ), 
    alpha = 0.2
  ) + 
  facet_wrap(vars(hb)) + 
  labs(
    # title = TeX("Mean $log_{10}$ income trajectories"),
    subtitle = "Salaried sales workers, no college degree",
    y = TeX("Estimated mean $log_{10}$ income"),
    x = "Age (years)",
    color = "",
    fill = ""
  ) + 
  ylim(c(4, 5.05)) + 
  theme(
    legend.position = c(0.9, 0.1)
  )
fhb_sales_plt
# 10^4.2 ~ 16k, 10^4.4 ~ 25k, 10^4.6 ~ 40k
ggsave(plot = fhb_sales_plt, filename = "fhb_sales_notitle.png" )


## salaried sales workers; college degree ----
fhb_sales_coll <- all_df %>% 
  filter(
    # female      ==   0,
    # hispanic    ==   0,
    # black       ==   0,
    college     ==   1,
    classworker ==   "Wage/salary",
    occ         ==   "sales"
  )


fhb_sales_coll_plt <- fhb_sales_coll %>% 
  ggplot() +
  geom_line(
    aes(
      y = Eyhat, x = age,
      color = gender,
    )
  ) + 
  geom_ribbon(
    aes(ymin = q.025, ymax = q.975, 
        x = age,
        # color = as_factor(female)
        fill = gender
    ), 
    alpha = 0.2
  ) + 
  facet_wrap(vars(hb)) + 
  labs(
    # title = TeX("Mean $log_{10}$ income trajectories"),
    subtitle = "Salaried sales workers with college degree",
    y = TeX("Estimated mean $log_{10}$ income"),
    x = "Age (years)",
    color = "",
    fill = ""
  ) + 
  ylim(c(4, 5.05)) +
  theme(
    legend.position = c(0.9, 0.1)
  )
fhb_sales_coll_plt

ggsave(plot = fhb_sales_coll_plt, filename = "fhb_sales_coll_plot_notitle.png" )



## salaried health workers; no college degree ---- 
health <- sal %>% filter(occ == "science", classworker == "Wage/salary")
table(health$female, health$black, health$college)
all_df %>% 
  filter(
    college     ==   1,
    classworker ==   "Wage/salary",
    occ         ==   "science"
  ) %>% 
  ggplot() +
  geom_line(
    aes(
      y = Eyhat, x = age,
      color = gender,
    )
  ) + 
  geom_ribbon(
    aes(ymin = q.025, ymax = q.975, 
        x = age,
        # color = as_factor(female)
        fill = gender
    ), 
    alpha = 0.2
  ) + 
  ylim(c(4, 5.1))+
  facet_wrap(vars(hb)) + 
  labs(
    # title = TeX("Mean $log_{10}$ income trajectories"),
    subtitle = "Salaried science workers, college degree",
    y = TeX("Estimated mean $log_{10}$ income"),
    x = "Age (years)",
    color = "",
    fill = ""
  ) + 
  theme(
    legend.position = c(0.9, 0.1)
  )
ggsave(file = "science_coll.png")




## VARSEL ----
kappas_sntc <- sim_res$kappa_sntc_mat[50,]
eta <- BFDR_eta_search(dropout_probs = kappas_sntc, max_rate = 0.05)
kappas_sntc < eta
round(kappas_sntc, 5)
# age                         female                       hispanic                          black 
# 0.00030                        0.00052                        0.00227                        0.00031 
# college classworkerGovernment employee       classworkerSelf-employed          occarchitect/engineer 
# 0.00028                        0.00050                        0.00034                        0.00085 
# occarts/sports/media         occbusiness operations              occcomputer/maths                occconstruction 
# 0.00079                        0.00271                        0.00131                        0.00337 
# occeducation                  occextraction                     occfarming                     occfinance 
# 0.00208                        0.00040                        0.00124                        0.00140 
# occfood                      occhealth                occinstallation                       occlegal 
# 0.00120                        0.00057                        0.00951                        0.00048 
# occmaintenance                  occmanagement               occpersonal care                  occproduction 
# 0.00125                        0.00034                        0.00020                        0.01027 
# occprotective                       occsales                     occscience              occsocial service 
# 0.00020                        0.00426                        0.00230                        0.01068 
# occtechnician              occtransportation 
# 0.00118                        0.00145






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










