##################################################
## Project:   demo plots for spline presentation
## Date:      Apr 12, 2019
## Author:    Arnie Seong
##################################################


# setup -------------------------------------------------------------------


library(splines)
library(RColorBrewer)
palramp <- colorRampPalette(c("deepskyblue", "purple", "hotpink"))
transp_pal <- function(pal, alpha=75){
  return( rgb(t(col2rgb(pal)), alpha = alpha, maxColorValue = 255) )
} 
pardefault <- par(no.readonly=T)





# functions ---------------------------------------------------------------


basis_mat <- function(x, ndx, bdeg=3, xrange, adj=0, want_knots = F){
  #modified from Eilers 1996
  #x = data vector; ndx = number intervals; bdeg = degree of spline polynomial
  #xl, xr (left and right of x-domain) optional
  #splineDesign() does the heavy lifting here (in splines package)
  #this wrapper just calculates knots at even intervals
  require(splines)
  if(missing(xrange)){ xrange <- c(min(x)-adj, max(x)+adj) }
  dx <- (xrange[2]-xrange[1]) / ndx
  knots <- seq(xrange[1]-bdeg*dx, xrange[2]+bdeg*dx, by=dx)
  #spline.des()$design returns basis matrix B
  B <- spline.des(knots=knots, x=x, ord=bdeg+1, derivs=0*x)$design
  
  if(want_knots==F){
    return(B)
  } else {
    return(list(B=B, knots=knots))
  }
}


make_Dk <- function(B, pen_ord=2){
  # K = order of penalty.  
  # k=0 gives ridge regression
  # penalty influences main diagonal and K subdiagonals
  # limit is a polynomial of order k-1 if degree of B-splines is >= k
  D <- diag(ncol(B))
  for (k in 1:pen_ord) D <- diff(D)
  return(D)
}


ps_regress <- function(B, y, Dk, lambda){
  Q <- solve(t(B)%*%B + lambda*t(Dk)%*%Dk)
  coefs <- Q%*%t(B)%*%y
  yhat <- B %*% coefs
  SS <- sum((y-yhat)^2)
  t1 <- sum(diag(Q%*%(t(B)%*%B)))
  gcv <- SS / ((nrow(B)-t1)^2)
  
  return(list(yhats=yhat, coefs=coefs, Q=Q, gcv=gcv))
}


choose_lambda<- function(B, y, Dk, lambdas=seq(0,2, .1), want_plot=T, return_lam=T){
  
  gcv_hold <- rep(NA, length(lambdas))
  for (i in 1:length(lambdas)){
    gcv_hold[i] <- ps_regress(B=B, y=y, Dk=Dk, lambda=lambdas[i])$gcv
  }
  best_lam <- lambdas[which(gcv_hold==min(gcv_hold))]
  print(paste0("best lambda = ", best_lam))
  
  if (want_plot){
    plot(x=lambdas, y=gcv_hold, type='l', lty=2)
    points(x=lambdas, y=gcv_hold)
  }
  
  if (return_lam){
    return(best_lam)
  }
}



# basis -------------------------------------------------------------------

# iterative construction of one basis
x <- seq(0,3, .05)

ind1 <- which(x<=1)
ind2 <- which(x>1 & x<=2 )
ind3 <- which(x>2 & x<=3 )

B1 <- x^2/2
B2 <- (-2*x^2 + 6*x -3)/2
B3 <- (3-x)^2/2

library(RColorBrewer)
pal <- brewer.pal(4, "Set2")
plot(B1~x, type="l", lwd=1, col=pal[1], ylim=c(0,2), ylab="B")
lines(B2~x, lwd=1, col=pal[2])
lines(B3~x, lwd=1, col=pal[3])


B <- c(B1[ind1], B2[ind2], B3[ind3])
lines(B~x, col=pal[4], lwd=2, lty=2)
legend("topleft", bty='n', lwd=1, col=pal, legend=c("P0", "P1", "P2", "B"))








#simple basis example: basis matrix, basis, plot with x points
x <- seq(-2, 2, .4)
n=length(x)
ndx <-  4
BK <- basis_mat(x, ndx, bdeg=3, want_knots=T)
B_s <- BK$B
# B_s
knots <- BK$knots


xlin <- seq(-2, 2, .1)
nlin = length(xlin)
BKlin <- basis_mat(xlin, ndx, bdeg=3, want_knots=T)
B_slin <- BKlin$B
# B_slin

matplot(y=B_s, x=x, type='l', lwd=2, col=brewer.pal(8, "Set2")
        , xlab="", ylab = "")
matlines(y=B_slin, x=xlin, type='l', lwd=2, col=transp_pal(brewer.pal(8, "Set2"), 80)
        , xlab="", ylab = "")
abline(v=knots, col="lightgrey", lty=2)
matpoints(x=x, y=B_s, col="grey50", pch=20)




# different degree bases

x <- seq(-2, 2, .01)
n=length(x)
ndx <-  8

#generate basis----
BK <- basis_mat(x, ndx, bdeg=1, want_knots=T)
B1 <- BK$B

BK <- basis_mat(x, ndx, bdeg=2, want_knots=T)
B2 <- BK$B

BK <- basis_mat(x, ndx, bdeg=3, want_knots=T)
B3 <- BK$B
knots <- BK$knots

pal <- palramp(2)


par(mfrow = c(3,1),
    oma = c(1,1,1,0) + 0.1,
    mar = c(1,0,2,2) + 0.1)
matplot(y=B1, x=x, type='l', lwd=2, col=brewer.pal(8, "Set2")
        , xlab="", ylab = "", yaxt="n", 
        main = "1st degree spline basis")
abline(v=knots, col="lightgrey", lty=2)

matplot(y=B2, x=x, type='l', lwd=2, col=brewer.pal(8, "Set2")
        , xlab="", ylab = "", yaxt="n", 
        main = "2nd degree spline basis")
abline(v=knots, col="lightgrey", lty=2)


matplot(y=B3, x=x, type='l', lwd=2, col=brewer.pal(8, "Set2")
        , xlab="", ylab = "", yaxt="n", 
        main = "3rd degree spline basis")
abline(v=knots, col="lightgrey", lty=2)
par(pardefault)





# zeroing out 4 coefficients -> 1 interval 0'd out (cubic splines) --------

x <- seq(-2, 2, .01)
n=length(x)
ndx <-  8

BK <- basis_mat(x, ndx, bdeg=3, want_knots=T)
B3 <- BK$B
knots <- BK$knots

dim(B3)
gamm_inds <- c(1:4, 9:11)
coef_test <- c(1,1,1,1,0,0,0,0,1,1,1)
coef_test2 <- c(.05,.1, .25, .15, 0,0,0,0,.15, .25, .5)

matplot(y=B3, x=x, type='l', lwd=2, col=brewer.pal(8, "Set2")
        , xlab="", ylab = "", yaxt="n", 
        main = "3rd degree spline basis",
        ylim=c(0,1.2))
lines(B3%*%coef_test~x, type='l')
lines(B3%*%coef_test2~x, type='l')
abline(v=knots, col="lightgrey", lty=2)







# spline fit --------------------------------------------------------------


#generate sine curve----
set.seed(31415)
x<-seq(1,10*pi, by=.3)
n=length(x)
y<-sin(x/5) + rnorm(length(x),0,.2)
ndx <-  10

#generate basis----
BK <- basis_mat(x, ndx, bdeg=3, want_knots=T)
B <- BK$B
knots <- BK$knots

#treating this like vanilla SLR----
beta_hat <- solve(t(B)%*%B)%*%t(B)%*%y
yhat <- B%*%beta_hat

#P-spline fit
Dk <- make_Dk(B=B)
lambda = choose_lambda(B, y, Dk, want_plot = F, return_lam=T)
ps_fit <- ps_regress(B, y, Dk, lambda)


# plotting----
# curve as sum of basis[i,]*coef[i]
coefs <- ps_fit$coefs
Bxcoef <- t(apply(B, 1, function(x) x*coefs)) 
pscurve <- rowSums(Bxcoef) 


Bxcoef_B <- t(apply(B, 1, function(x) x*beta_hat)) 

Bxpal <- transp_pal("hotpink", 75)
# plot B-spline fit
plot(y ~ x, pch = 20, cex=.75, col="grey", 
     main="B-spline fit to sine curve + noise")
abline(v=knots, col="lightgrey", lty=2)  # knots
matpoints(x=x, y=B, type='l', lty=3, col='grey') # basis
matpoints(x=x, y=Bxcoef_B, type='l', lty=1, col=Bxpal, lwd=1.5) #coefs*basis
points(y=yhat, x=x, col="red", type='l') # B-spline fit
legend("topright", bty='n', 
       legend = c("B-spline fit", "B-spline coef*basis", "basis"), 
       lty=c(1, 1, 1, 2), lwd=2, col=c("red", Bxpal, "grey"))



#plot P-spline and B-spline fit
plot(y ~ x, pch = 20, cex=.75, col="grey", 
     main="P-spline fit to sine curve + noise")
abline(v=knots, col="lightgrey", lty=2)  # knots
matpoints(x=x, y=B, type='l', lty=3, col='grey') # basis
matpoints(x=x, y=Bxcoef_B, type='l', lty=1, col=Bxpal, lwd=1.5) #coefs*basis for B-spline
matpoints(x=x, y=Bxcoef, type='l', lty=1, col="lightblue", lwd=1.5) #coefs*basis ofr P-spline
points(y=yhat, x=x, col="red", type='l') # B-spline fit
points(pscurve~x, type='l', col="blue") # P-spline fit
legend("topright", bty='n', 
       legend = c("P-spline fit", "B-spline fit", "P coef*basis", "B coef*basis", "basis"), 
       lty=c(1, 1, 1, 1, 2), lwd=2, col=c("blue", "red", "lightblue", Bxpal, "grey"))








# spline fit categorical --------------------------------------------------


#generate data curve----
set.seed(31415)
x<-seq(1,10*pi, by=.3)
n=length(x)
y<-sin(x/5) + rnorm(length(x),0,.2)
cl <- sample(size=n, c(rep(0,floor(n/2)), rep(1,floor(n/2+1)))) #randomly assign classes
ndx <-  10

#generate basis----
BK <- basis_mat(x, ndx, bdeg=3, want_knots=T)
B <- BK$B
B2 <- cbind(B, B)
knots <- BK$knots

#concatenating basis and female basis
M_Bcol <- dim(B)[2]
BF <- apply(B, 2, function(x) x*cl)
B <- cbind(B, BF)
#treating this like vanilla SLR----
beta_hat <- solve(t(B)%*%B)%*%t(B)%*%y
yhat <- B%*%beta_hat

#P-spline fit
Dk <- make_Dk(B=B)
lambda = choose_lambda(B, y, Dk, want_plot = F, return_lam=T)
ps_fit <- ps_regress(B, y, Dk, lambda)


# plotting----
# curve as sum of basis[i,]*coef[i]
coefs <- ps_fit$coefs
Bxcoef <- t(apply(B, 1, function(x) x*coefs)) 
F_inds <- which(cl==1)
M_inds <- setdiff(c(1:n), F_inds)
pscurve <- rowSums(Bxcoef) 

M_Bxcoef <- Bxcoef[,1:M_Bcol]
F_Bxcoef <- Bxcoef[,1:M_Bcol] + 
  t(apply(Bxcoef[,1:M_Bcol], 1, function(x) x*coefs[(M_Bcol+1):dim(B)[2]])) 

pal <- palramp(2)
pal_B <- palramp(5)[c(2,4)]
pal_transp <- transp_pal(pal, alpha = 75)
grey_transp <- transp_pal('grey', alpha = 200)
palette(pal_transp)
plot(y ~ x, pch = 20, cex=.75, col=cl+1, 
     main="spline fit to sine curve + noise; randomly assigned classes")
abline(v=knots, col=grey_transp, lty=2)  # knots
matpoints(x=x, y=B[, 1:M_Bcol], type='l', lty=3, col=grey_transp) # basis
matpoints(x=x, y=M_Bxcoef, type='l', lty=1, col=pal_transp[1]) #coefs*basis
matpoints(x=x, y=F_Bxcoef, type='l', lty=1, col=pal_transp[2]) #coefs*basis
points(pscurve[M_inds]~x[M_inds], type='l', col=pal[1]) # Male P-spline fit
points(pscurve[F_inds]~x[F_inds], type='l', col=pal[2]) # Female P-spline fit
# points(yhat[M_inds]~x[M_inds], type='l', col=pal_B[1]) # Male B-spline fit
# points(yhat[F_inds]~x[F_inds], type='l', col=pal_B[2]) # Female B-spline fit
legend("topright", bty='n', 
       legend = c("Class 1", "Class 2"), 
       lty=1, lwd=2, col=pal)


