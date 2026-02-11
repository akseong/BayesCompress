##################################################
## Project:   sparseVCBART data
## Date:      Feb 10, 2026
## Author:    Arnie Seong
##################################################


beta_0 <- function(z1, z2){
  3*z1 + 
    (sin(pi * z1)) * (2 - 5 * (z2 > 0.5)) - 
    2 * (z2 > 0.5)
}

beta_1 <- function(z1){
  (3 - 3*z1^2)*(z1 > 0.6) - 10*sqrt(z1)*(z1 < 0.25)
}

beta_2 <- 1

beta_3 <- function(z1, z2, z3, z4, z5){
  10*sin(pi*z1*z2) + 20*((z3-0.5)^2) + 10*z4 + 5*z5
}

# R = 20, p = 3




