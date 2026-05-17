

# setup -------------------------------------------------------------------


library(splines)
library(RColorBrewer)
palramp <- colorRampPalette(c("deepskyblue", "purple", "hotpink"))
transp_pal <- function(pal, alpha=75){
  return( rgb(t(col2rgb(pal)), alpha = alpha, maxColorValue = 255) )
} 
pardefault <- par(no.readonly=T)

vismat <- function(mat, cap = NULL, leg = TRUE, na0 = TRUE, lims = NULL, square = NULL, preserve_rownums = TRUE){
  # outputs visualization of matrix with few unique values
  # colnames should be strings, values represented as factors
  # sci_not=TRUE puts legend in scientific notation
  require(ggplot2)
  require(scales)
  require(reshape2)
  
  if(!preserve_rownums) rownames(mat) <- NULL
  
  melted <- melt(mat)
  melted$value <- ifelse(
    melted$value == 0 & na0,
    NA,
    melted$value
  )
  p <- ggplot(melted) + 
    geom_raster(aes(y = Var1, 
                    x = Var2, 
                    fill = value)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    scale_fill_viridis_c(limits = lims) + 
    scale_x_discrete(expand = c(0,0))
  
  if (is.numeric(melted$Var1)){
    p <- p + 
      scale_y_reverse(expand = c(0,0))
  } else {
    p <- p + 
      scale_y_discrete(limits = rev(levels(melted$Var1)), expand = c(0,0))
  }
  
  
  
  if (is.null(square)) square <- nrow(mat) / ncol(mat) > .9 & nrow(mat) / ncol(mat) < 1.1
  if (square) p <- p + coord_fixed(1)
  
  if(is.null(cap)) cap <- paste0("visualization of matrix ", substitute(mat))
  
  p <- p + labs(title=cap)
  
  if (!leg) p <- p + theme(legend.position = "none")
  
  return(p)
}



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


B_s

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



###--------------------------

x <- seq(-2, 2, .1)
n=length(x)
ndx <-  8
x2 <- rep(x, 2)
BK <- basis_mat(x2, ndx, bdeg=3, want_knots=T)
B0 <- BK$B
BK <- basis_mat(x2, ndx, bdeg=0, want_knots=T)
B1_base <- BK$B
ind <- rep(c(1, 0), each = length(x))
B1_raw <- sweep(B1_base, 1, ind, "*")

orthog_bases <- function(B0, B1_raw, roundoff_0s=FALSE, tol=1e-12){
  # orthogonalize B1_raw against B0
  BtB0 <- t(B0)%*%B0
  B1_orth <- B1_raw - (B0 %*% solve(BtB0)) %*% (t(B0)%*%B1_raw)
  
  # turn floating point errors back into 0's (not usually needed)
  if (roundoff_0s){
    B1_orth <- matrix(
      ifelse(abs(B1_orth) < tol, 0, B1_orth),
      nrow=nrow(B1_orth),
      ncol=ncol(B1_orth)
    )
  }
  
  return(B1_orth)
}

W1 <- orthog_bases(B0, B1_raw)
summary(abs(W1))
X <- cbind(B0, W1)

WTWinv <- solve(t(X) %*% X)
WTWinvWT <- WTWinv %*% t(X)

threshold_0 <- function(y, thresh = 1e-5){
  tf_vec <- abs(y) < thresh
  y[tf_vec] <- 0
  return(y)
}



vismat(WTWinvWT)

vismat(X)

vismat(WTWinv)


vismat(cbind(B0, B1_raw))


