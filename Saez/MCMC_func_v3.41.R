##################################################
## Project:   MCMC_func_v3.41
## Date:      Aug 15, 2020
## Author:    Arnie Seong
##################################################


####              #### ----
####    BASICS    #### ----
####              #### ----
# loop_time(loopindex, st_time, every=10, round_dig=4, overall_st_time)----
# place "st_time <- loop_time(j, st_time, every=10)" as last line of loop
loop_time <- function(loopindex, st_time, every=10, round_dig=4, overall_st_time){
  if(loopindex!=1 &&  loopindex%%every==0) {
    cat("iter ", loopindex ,": ", 
        round(Sys.time()-st_time, round_dig), ";  ")
    
    if(!missing(overall_st_time)){
      cat("total time: ", round(Sys.time()-overall_st_time, round_dig), "  \n  ")
    }
    
    cat("  \n  ")
  }
  if(loopindex==1 || loopindex%%every==0) st_time <- Sys.time()
  return(st_time)
}



# parent_func(n_calls=2) ----
#place parent2 <- parent_func() as first line in current function
#parent2 is the name of the function that called current function
parent_func <- function(n_calls=2) deparse(sys.call(-n_calls))


# palramp(c("deepskyblue", "purple", "orangered"))----
palramp <- colorRampPalette(c("deepskyblue", "purple", "orangered"))



# transp_pal((pal, alpha=75))----
transp_pal <- function(pal, alpha=75){
  return( rgb(t(col2rgb(pal)), alpha = alpha, maxColorValue = 255) )
} 







# vismat_factor(mat, cap = NULL, leg = T, sci_not=T, na0=T)----
vismat_factor <- function(mat, cap = NULL, leg = TRUE, sci_not = TRUE, na0 = TRUE, square){
  # outputs visualization of matrix with few unique values
  # colnames should be strings, values represented as factors
  # sci_not=TRUE puts legend in scientific notation
  require(ggplot2)
  require(scales)
  require(reshape2)
  
  melted <- melt(mat)
  melted$value <- ifelse(
    melted$value == 0 & na0,
    NA,
    melted$value
  )
  p <- ggplot(melted) + 
    geom_raster(aes(y = Var1, 
                    x = Var2, 
                    fill = factor(value))) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  if (is.numeric(melted$Var1)){
    p <- p + 
      scale_y_reverse()
  } else {
    p <- p + 
      scale_y_discrete(limits = rev(levels(melted$Var1)))
  }
  
  if (sci_not) {
    p <- p + 
      scale_fill_viridis_d(labels = scientific(as.numeric(levels(as.factor(melted$value)))))
  } else {
    p <- p + scale_fill_viridis_d()
  }
  
  if (missing(square)) square <- nrow(mat) / ncol(mat) > .9 & nrow(mat) / ncol(mat) < 1.1
  if (square) p <- p + coord_fixed(1)
  
  if (!is.null(cap)) p <- p + labs(title=cap)
  
  if (!leg) p <- p + theme(legend.position = "none")
  
  return(p)
}

# vismat_cut(mat, brks=NULL, n_brks=10, cap=NULL, leg=T, sci_not=T, na0 = TRUE, square)----
vismat_cut <- function(mat, brks=NULL, n_brks = 10, cap = NULL, leg = TRUE, sci_not = TRUE, na0 = TRUE, square){
  # outputs visualization of matrix with values cut into quantiles
  # colnames should be strings, values represented as factors
  # sci_not=TRUE puts legend in scientific notation
  require(ggplot2)
  require(scales)
  require(reshape2)
  
  melted <- melt(mat)
  melted$value <- ifelse(
    melted$value == 0 & na0,
    NA,
    melted$value
  )
  brks_provided <- ifelse(is.null(brks), F, T)
  if (is.null(brks)){
    brks <- unique(
      quantile(
        melted$value, 
        probs = seq(0, 1, length.out=n_brks), 
        na.rm = TRUE
        )
      )
  }
  melted$qtile <- cut(melted$value, breaks = brks, include.lowest = TRUE)
  p <- ggplot(melted) + 
    geom_raster(aes(y = Var1, 
                    x = Var2, 
                    fill = qtile)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  if (is.numeric(melted$Var1)){
    p <- p + 
      scale_y_reverse()
  } else {
    p <- p + 
      scale_y_discrete(limits = rev(levels(melted$Var1)))
  }
  
  if (sci_not) {
    brks <- scientific(brks)
    nlev <- length(brks)
    lvls <- paste0(brks[1:(nlev-1)], ":", brks[2:nlev])
    p <- p + 
      scale_fill_viridis_d(labels = lvls)
  } else {
    p <- p + scale_fill_viridis_d()
  }
  
  if (missing(square)) square <- nrow(mat) / ncol(mat) > .9 & nrow(mat) / ncol(mat) < 1.1
  if (square) p <- p + coord_fixed(1)
  
  if (!is.null(cap)) p <- p + labs(title=cap)
  
  if (!leg) p <- p + theme(legend.position = "none")
  
  if (brks_provided) p <- p + labs(fill = "ranges")
  
  return(p)
}

# vismat(mat, cap = NULL, leg = TRUE, na0 = TRUE, square)----
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



####                                     #### ---- 
####    data generation/visualization    #### ----
####                                     #### ---- 

# gen_dat(n_obs, eta, sig, ndz, ...)----
gen_dat <- function(n_obs, eta, sig, ndz, 
                    interv0 = c(15, 25), zrange=c(0,40), bdeg=3, 
                    wantplot=F, pal=c("deepskyblue", "orangered")){
  # length(eta) must be 2*(ndz+bdeg)
  x1 <- sample(c(0,1), n_obs, prob=c(.6, .4), replace=T)
  zl <- zrange[1]
  zr <- zrange[2]
  z <- runif(n_obs, zl, zr)
  eps <- rnorm(n_obs, 0, sig)
  x1_inds <- which(x1==1)
  
  bk <- basis_mat(x=z, ndx=ndz, bdeg=bdeg, xl=zl, xr=zr)
  knots <- bk$all_knots
  bmat <- bk$B
  bmat_cnames <- paste0("[", knots[1:(dim(bmat)[2])], 
                        ":", knots[1:(dim(bmat)[2])+(bdeg+1)], 
                        "]") #column names indicate curve's active interval
  colnames(bmat) <- bmat_cnames
  bmat_x1 <- bmat*x1
  colnames(bmat_x1) <- paste0(bmat_cnames, "_x1")
  bmat_all <- cbind(bmat, bmat_x1)
  
  # identify eta_j's corresponding to interval to 0 out
  names(eta) <- colnames(bmat_all) 
  ind0_left <- min(which(knots>=interv0[1]))-bdeg + dim(bmat)[2]
  ind0_right <- max(which(knots<=interv0[2]))-1 + dim(bmat)[2]
  eta0 <- eta
  eta0[ind0_left:ind0_right] <- 0
  
  eps <- rnorm(n_obs, 0, sig)
  y <- bmat_all%*%eta + eps
  y0 <- bmat_all%*%eta0 + eps
  
  return(list("y"=y, "y0"=y0, "z"=z, "x1"=x1, # data
              "knots"=knots,
              "B"=bmat_all,                   # B
              "B_trunc"=bmat_all[,-(ind0_left:ind0_right)],
              "eta"=eta,
              "eta0"=eta0,
              "B0cols"=ind0_left:ind0_right,   # eta_j's to 0 out
              "zrange"=zrange,   # for pplotting (vis_dat function)
              "ndz"=ndz,
              "interv0"=interv0,
              "bdeg"=bdeg))
}




# vis_dat(dat, want_grids=F, which_plots=c("full","reduced"))----
vis_dat <- function(dat, pal=c("deepskyblue", "orangered"), 
                    want_grids=F, which_plots=c("full","reduced"), 
                    want_bases=T, want_plots=T, transp_alph=75){
  pardefault <- par(no.readonly=T)
  transp_pal <- function(pal, alpha=75){
    return( rgb(t(col2rgb(pal)), alpha = alpha, maxColorValue = 255) )
  } 
  pal_basis <- pal
  pal_dat <- transp_pal(pal, transp_alph)
  
  
  zl <- dat$zrange[1]  
  zr <- dat$zrange[2]
  zgrid <- seq(zl, zr, (zr-zl)/100)
  Bgrid <- basis_mat(x=zgrid, ndx=dat$ndz, bdeg=dat$bdeg, 
                     xl=zl, xr=zr, want_knots=F)
  Bgridx2 <- cbind(Bgrid, Bgrid)
  
  ylims <- c(min(dat$y, dat$y0), max(dat$y, dat$y0))
  
  
  yhat_cat0 <- Bgrid%*%dat$eta[1:(dat$ndz+dat$bdeg)]
  yhat_cat1 <- Bgridx2%*%dat$eta
  bxeta <- t(apply(Bgridx2, 1, function(x) x*dat$eta))
  
  
  yhat0_cat0 <- Bgrid%*%dat$eta0[1:(dat$ndz+dat$bdeg)]
  yhat0_cat1 <- Bgridx2%*%dat$eta0
  bxeta0 <- t(apply(Bgridx2, 1, function(x) x*dat$eta0))
  
  if(want_plots){
    palette(pal_dat)
    # plot full model
    if (is.element("full", which_plots)){
      plot(dat$y~dat$z, ylim=ylims, pch=20, col=dat$x1+1, 
           main=paste0(dat$ndz + dat$bdeg,
                       " basis curves with no 0 interval"))
      if (want_bases){
        matlines(y=bxeta[,1:(dat$ndz+dat$bdeg)], x=zgrid, lty=3, col=pal_basis[1])
        matlines(y=bxeta[,(dat$ndz+dat$bdeg+1):dim(bxeta)[2]], x=zgrid, lty=3, col=pal_basis[2])
      }
      lines(yhat_cat0~zgrid, col=pal[1], lwd=2)
      lines(yhat_cat1~zgrid, col=pal[2], lwd=2)
      abline(v=dat$interv0, lty=2, col="grey")
    }
    
    #plot reduced model
    if (is.element("reduced", which_plots)){
      plot(dat$y0~dat$z,  ylim=ylims, pch=20, col=dat$x1+1, 
           main=paste0(dat$ndz + dat$bdeg," basis curves with 0 interval (", 
                       dat$interv0[1], ", ", dat$interv0[2], ")"))
      if (want_bases){
        matlines(y=bxeta0[,1:(dat$ndz+dat$bdeg)], x=zgrid, lty=3, col=pal_basis[1])
        matlines(y=bxeta0[,(dat$ndz+dat$bdeg+1):dim(bxeta0)[2]], x=zgrid, lty=3, col=pal_basis[2])
      }
      lines(yhat0_cat0~zgrid, col=pal[1], lwd=2)
      lines(yhat0_cat1~zgrid, col=pal[2], lwd=2)
      abline(v=dat$interv0, lty=2, col="grey")
    }
    par(pardefault)
  }
  if (want_grids==T){
    return(list("zgrid"=zgrid,
                "Bgrid"=Bgrid,
                "Bgridx2"=Bgridx2,
                "Bxeta"=bxeta,
                "Bxeta0"=bxeta0,
                "yhat_cat0" = yhat_cat0,
                "yhat_cat1" = yhat_cat1,
                "yhat0_cat0" = yhat0_cat0,
                "yhat0_cat1" = yhat0_cat1))
  }
}



# vis_fit(dat, y, eta_true,etahat, ndz_fit, ...)----
vis_fit <- function(dat, y, eta_true, #truth
                    etahat, ndz_fit, bdeg_fit=3, zl_f, zr_f, interv0, #data
                    pal=c("deepskyblue", "orangered"), transp_alph=75,
                    show_bases=T, maintext){
  pardefault <- par(no.readonly=T)
  transp_pal <- function(pal, alpha=75){
    return( rgb(t(col2rgb(pal)), alpha = alpha, maxColorValue = 255) )
  } 
  pal_basis <- pal
  pal_dat <- transp_pal(pal, transp_alph)
  
  #if not supplied, assume same parameters used
  if(missing(interv0)){ interv0 <- dat$interv0}
  if(missing(ndz_fit)){ ndz_fit <- length(etahat)/2-bdeg_fit}
  if(missing(zl_f)){zl_f <- range(dat$z)[1]}
  if(missing(zr_f)){zr_f <- range(dat$z)[2]}
  
  #truth
  zl <- dat$zrange[1]  
  zr <- dat$zrange[2]
  zgrid <- seq(zl, zr, (zr-zl)/100)
  Bgrid <- basis_mat(x=zgrid, ndx=dat$ndz, bdeg=dat$bdeg, 
                     xl=zl, xr=zr, want_knots=F)
  Bgridx2 <- cbind(Bgrid, Bgrid)
  
  #fit
  
  zgrid_f <- seq(zl_f, zr_f, (zr_f-zl_f)/100)
  Bgrid_f <- basis_mat(x=zgrid_f, ndx=ndz_fit, bdeg=bdeg_fit, 
                       xl=zl_f, xr=zr_f, want_knots=F)
  Bgridx2_f <- cbind(Bgrid_f, Bgrid_f)
  
  
  ylims <- range(dat$y, dat$y0)
  
  palette(pal_dat)
  
  if (missing(maintext)){
    maintext<-paste0(ndz_fit + bdeg_fit," basis curves; fitted and truth")
  }
  
  plot(y~dat$z, ylim=ylims, pch=20, col=dat$x1+1, 
       main=maintext)
  
  #plot basis functions * etahat
  if (show_bases==T){
    bxeta_f <- t(apply(Bgridx2_f, 1, function(x) x*etahat))
    matlines(y=bxeta_f[,1:(ndz_fit+bdeg_fit)], x=zgrid_f, lty=3, col=pal_basis[1])
    matlines(y=bxeta_f[,(ndz_fit+bdeg_fit+1):dim(bxeta_f)[2]], x=zgrid_f, lty=3, col=pal_basis[2])
  }
  
  #plot true function
  y_cat0 <- Bgrid%*%eta_true[1:(dat$ndz+dat$bdeg)]
  y_cat1 <- Bgridx2%*%eta_true
  lines(y_cat0~zgrid, col=pal[1], lwd=2, lty=2)
  lines(y_cat1~zgrid, col=pal[2], lwd=2, lty=2)
  
  #plot estimates
  yhat_cat0 <- Bgrid_f%*%etahat[1:(ndz_fit+bdeg_fit)]
  yhat_cat1 <- Bgridx2_f%*%etahat
  lines(yhat_cat0~zgrid_f, col=pal[1], lwd=2, lty=1)
  lines(yhat_cat1~zgrid_f, col=pal[2], lwd=2, lty=1)
  abline(v=interv0, lty=2, col="grey")
  legend("topright", bty="n", lty=c(2,1,2,1), lwd=2, col=pal[c(1,1,2,2)],
         legend=c("true cat.0", "fitted cat.0", "true cat.1", "fitted cat.1"))
  par(pardefault)
}




# gen_eta(ndz=16, bdeg=3, keepdir_steps=7)----
gen_eta <- function(ndz=16, bdeg=3, keepdir_steps=7){
  # enter "NA" for keepdir_steps to randomly choose between 4 and 10
  eta_len <- 2*(ndz+bdeg)
  eta <- rep(NA, eta_len)
  
  if(is.na(keepdir_steps)){
    keepdir_steps <- round(runif(1,4,10))
  }
  
  lo=0
  pm=1
  for (i in 1:(eta_len)){
    # make curve keep same direction
    # for keepdir_steps
    if (i%%keepdir_steps == 0) { 
      pm <- sample(c(-1,1), 1, prob=c(.5, .5)) 
      lo=0
    }
    lo <- lo+pm*abs(rnorm(1,0,1))
    eta[i] <- round( runif(1, lo-.2, lo+.5), 2 )
  }
  return(eta)
}


# gen_eta2(ndz=16, bdeg=3, keepdir_steps=7)----
#less wiggly data:
gen_eta2 <- function(ndz=16, bdeg=3, keepdir_steps=7){
  # enter "NA" for keepdir_steps to randomly choose between 4 and 10
  eta_len <- 2*(ndz+bdeg)
  eta <- rep(NA, eta_len)
  
  if(is.na(keepdir_steps)){
    keepdir_steps <- round(runif(1,4,10))
  }
  
  eta_i=0
  pm=1
  for (i in 1:(eta_len)){
    # make curve keep same direction
    # for keepdir_steps
    if (i%%keepdir_steps == 0) { 
      pm <- sample(c(-1,1), 1, prob=c(.5, .5)) 
    } else if (i%%keepdir_steps == round(keepdir_steps/2)) {
      pm=-pm
    }
    
    eta_i <- eta_i+pm*abs(rnorm(1,0,1))
    eta[i] <- round( runif(1, eta_i-.5, eta_i+.5), 2 )
  }
  return(eta)
}



# insert0(eta0, eta, B0cols)----
insert0 <- function(eta0, eta, B0cols){
  # put coerced 0's back into coefficients returned by fit0
  trunc_start <- min(B0cols)
  trunc_end <- max(B0cols)+1
  eta0_end <- eta0[trunc_start:length(eta0)]
  eta0_new <- rep(NA, length(eta))
  names(eta0_new) <- colnames(eta) 
  eta0_new[1:trunc_start] <- eta0[1:trunc_start]
  eta0_new[trunc_start:trunc_end] <- 0
  eta0_new[trunc_end:length(eta0_new)] <- eta0_end
  return(eta0_new)
}



# fit_bmats(z, ndz, x1, ...)----
fit_bmats <- function(z, ndz, x1, interv0=c(15, 25), zrange=range(z), bdeg=3, roundk_dig=2){
  zl <- zrange[1]
  zr <- zrange[2]
  bk <- basis_mat(x=z, ndx=ndz, bdeg=bdeg, xl=zl, xr=zr)
  
  #create design matrices
  knots <- round(bk$all_knots, roundk_dig)
  bmat <- bk$B
  bmat_cnames <- paste0("[", knots[1:(dim(bmat)[2])], 
                        ":", knots[1:(dim(bmat)[2])+(bdeg+1)], 
                        "]") #column names indicate curve's active interval
  colnames(bmat) <- bmat_cnames
  bmat_x1 <- bmat*x1
  colnames(bmat_x1) <- paste0(bmat_cnames, "_x1")
  bmat_all <- cbind(bmat, bmat_x1)
  
  #  identify 0'd intervals
  ind0_left <- min(which(knots>=interv0[1]))-bdeg + dim(bmat)[2]
  ind0_right <- max(which(knots<=interv0[2]))-1 + dim(bmat)[2]
  
  return(list("z"=z,
              "ndz"=ndz,
              "bdeg"=bdeg,
              "interv0"=interv0,
              "zrange"=zrange,
              "B0cols"=ind0_left:ind0_right,
              "B_cnames"=colnames(bmat_all),
              "B"=bmat_all,
              "B_trunc"=bmat_all[,-(ind0_left:ind0_right)]))
}



# gen_dat_smooth ----
gen_dat_smooth <- function(n_obs=200, sig=2, 
                           interv0=c(15,25), zrange=c(0,40),
                           min_cuts=4, max_cuts=8, # number pieces 
                           max_coef=2, #lower values encourage less up/down travel
                           max_range=20, #max difference between max/min of each curve
                           max_diff=5, 
                           want_sep=T, #T encourages further separation of curves
                           min_diff_interv0=3, #control difference in the 0 interval 
                           smooth_span=.3, #loess smoother span
                           want_plot=F, want_raw_plot=F, 
                           rand_return=F, #randomize data by row? otherwise split by indicator
                           pal=c("deepskyblue", "orangered", "violet"), 
                           transp_alph=50, dottype=c(20, 20, 19),
                           lwidth=c(1,1,2), ltype=c(2,2,2)){
  # construct z1 and z2 (2 diff datasets)
  zmin <- zrange[1]
  zmax <- zrange[2]
  z <- seq(zmin, zmax, (zmax-zmin)/(n_obs*5))
  
  #draw number of cuts to make in function
  n_cuts1 <- runif(1, min_cuts, max_cuts)
  n_cuts2 <- runif(1, min_cuts, max_cuts)
  
  cuts1 <- c(zmin, sample(z, n_cuts1), zmax)
  cuts1 <- cuts1[order(cuts1)]
  cuts2 <- c(zmin, sample(z, n_cuts2), zmax)
  cuts2 <- cuts2[order(cuts2)]
  
  #random coefficients  
  pm1 <- sample(c(-1,1), length(cuts1), replace=T)
  pm2 <- sample(c(-1,1), length(cuts2), replace=T)
  coefs1 <- pm1*runif(length(cuts1), .25, max_coef)
  coefs2 <- pm2*runif(length(cuts2), .5, max_coef)
  
  #powers for polynomial pieces
  avail_pows <- c(.25, .5, .75, 2, 3)
  pows1 <- sample(avail_pows, length(cuts1), replace=T)
  pows2 <- sample(avail_pows[1:3], length(cuts2), replace=T)
  
  #put into lists to generate unsmoothed functions in loop
  cutlist <- list(cuts1, cuts2)
  coeflist <- list(coefs1, coefs2)
  powlist <- list(pows1, pows2)
  ylist <- list(NULL, NULL)
  # y_smlist <- list(NULL, NULL)
  if(missing(max_diff)) max_diff <- max_range/2
  rangelist <- list(max_range/2, max_diff)
  
  ## generate unsmoothed functions ##
  for(l in 1:2){
    cuts <- cutlist[[l]]
    coefs <- coeflist[[l]]
    pows <- powlist[[l]]
    
    y=NULL
    y_temp_end=0
    
    for(i in 1:(length(cuts)-1)){
      st_ind <- which(z==cuts[i])
      end_ind <- which(z==cuts[i+1])
      interv <- z[st_ind:end_ind]
      
      #avoid duplicating endpoints across loops
      # n_interv <- ifelse(i==(length(cuts)-1), 
      #                    length(interv), 
      #                    length(interv)-1)
      # actually, duplicating is fine.  Just cut down to size later
      n_interv <- length(interv)
      
      mid_interv <- sample(interv,1)
      #avoid NaN's (negative numbers to partial powers) 
      if (pows[i] < 1) mid_interv <- max(interv) 
      
      #polynomial piece*coef
      if(want_sep==F){ coefs[i] <- coefs[i]/pows[i] }
      y_temp <- (coefs[i]
                 *(mid_interv-interv)^pows[i])
      y_temp <- y_temp + (y_temp_end-y_temp[1])
      
      y <- c(y, y_temp[1:n_interv])
      y_temp_end <- y_temp[n_interv]
    }
    if (l==2) y <- y - min(abs(prod(coeflist[[l]])), 4)
    #scale
    sc <- rangelist[[1]]/diff(range(y))
    y<- y*sc 
    ylist[[l]] <- y[1:length(z)] #cut down to correct size
  }
  
  ## generate data using loess-smoothed functions ##
  #smooth out using loess
  y2 <- (ylist[[1]] + ylist[[2]])
  y1 <- ylist[[1]]
  int0_st <- max(which(z < interv0[1]))
  int0_end <- min(which(z > interv0[2]))
  
  #ensure interv0 is different between the 2 by at least min_diff_interv0
  int0_diffs <- y2[int0_st:int0_end]-y1[int0_st:int0_end]
  if(min(abs(int0_diffs)) < min_diff_interv0){
    y2[int0_st:int0_end] <- sign(int0_diffs)*min_diff_interv0 + y2[int0_st:int0_end]
  }
  
  
  #adjust for smoother's span to create 0 interval in y0
  n_z <- length(z)
  span_adj <- round(length(z)*smooth_span/2.75)+1
  int0_st <- max(int0_st-span_adj,0)
  int0_end <- min(int0_end+span_adj, length(z))
  
  y20 <- c(y2[1:(int0_st-1)],
           y1[int0_st:(int0_end-1)],
           y2[int0_end:length(z)])
  
  y1_sm <- loess(y1~z, span=smooth_span)$fitted
  y2_sm <- loess(y2~z, span=smooth_span)$fitted
  y20_sm <- loess(y20~z, span=smooth_span)$fitted
  
  ## generate data ##
  #generate indicator variable
  x1 <- sample( c(0,1), n_obs, replace=T)
  #sample of z-grid to generate data
  obs_inds <- sample(c(1:length(z)), n_obs, replace=T)
  
  #separate observations by indicator
  m_inds <- obs_inds[which(x1==0)]
  f_inds <- obs_inds[which(x1==1)]
  
  #generate errors
  m_eps <- rnorm(length(m_inds), 0, sig)
  f_eps <- rnorm(length(f_inds), 0, sig)
  
  z_m <- z[m_inds]
  z_f <- z[f_inds]
  y_m <- y1_sm[m_inds] + m_eps
  y_f <- y2_sm[f_inds] + f_eps
  y_f0 <- y20_sm[f_inds] + f_eps
  
  
  #stack and return
  x1_out <- c(rep(0,length(m_inds)), 
              rep(1, length(f_inds)))
  z_out <- c(z_m, z_f)
  y_out <- c(y_m, y_f)
  y0_out <- c(y_m, y_f0)
  
  #randomize (so dataset not split by indicator)
  if(rand_return){
    r_inds <- sample(1:n_obs, n_obs)
    x1_out <- x1_out[r_inds]
    z_out <- z_out[r_inds]
    y_out <- y_out[r_inds]
    y0_out <- y0_out[r_inds]
  }
  
  ## plotting ##
  #plot control 
  pardefault <- par(no.readonly=T)
  if(want_raw_plot && want_plot){
    par(mfrow = c(2,1),
        oma = c(2,2,1,0) + 0.1,
        mar = c(0,0,0,1) + 0.1)
  }
  
  #plot loess-smoothed function + data generated
  if(want_plot){
    paltransp <- transp_pal(pal, alpha=transp_alph)
    plot( y2_sm~z, col=pal[2], type='l',
          ylim=range(y_out), lwd=lwidth[2], lty=ltype[2])
    lines(y1_sm~z, col=pal[[1]], lwd=lwidth[1], lty=ltype[1])
    lines(y20_sm~z, col=pal[[3]], lwd=lwidth[3], lty=ltype[3])
    points(y_f ~ z_f, col=paltransp[2], pch=dottype[2])
    points(y_m ~ z_m, col=paltransp[1], pch=dottype[1])  
    points(y_f0 ~ z_f, col=paltransp[3], pch=dottype[3])
    abline(v=interv0, col="grey", lty=2)
    legend("topleft", col=pal, lty=ltype, lwd=lwidth, bty='n', 
           legend=c("ind=0", "ind=1", "ind=1 coerced 0"))
  }
  
  #plot unsmoothed functions
  if(want_raw_plot){
    paltransp <- transp_pal(pal, alpha=transp_alph)
    plot( y2~z, col=pal[2], type='l',
          ylim=range(y_out))
    lines(y1~z, col=pal[[1]])
    abline(v=interv0, col="grey", lty=2)
  }
  par(pardefault)
  
  return(list("y"= y_out, 
              "y0"=y0_out, 
              "z"=z_out, 
              "x1"=x1_out,
              "interv0"=interv0)
  )  
}




#### FOR SAEZ DATA ----
# load_Saez(patient_num, elec_num, timings_w=seq(-900, 1900, 50))----
load_Saez <- function(patient_num, elec_num, timings_w=seq(-900, 1900, 50), path_to_Rdata){
  if (!missing(path_to_Rdata)){
    load(path_to_Rdata)
  } else {
    load("ofc3_prewindowed.RData")
  }
  
  pt_gamble_data <- gamble_data %>% filter(patient==patient_num)
  pt_gamble_data$trial <- 1:nrow(pt_gamble_data)
  pt_ecog <- buttonpressw[[patient_num]]
  pt_elec_dat <- pt_ecog[,,elec_num]
  
  # formatting so data is all together
  library(reshape2)
  df_hfa <- melt(t(pt_elec_dat))
  names(df_hfa) <- c("time_index", "trial", "hfa")
  df_hfa$timings_w <- rep(timings_w, nrow(pt_gamble_data))
  pt_gamble_data$trial <- 1:nrow(pt_gamble_data)
  pt_bigdf <- inner_join(pt_gamble_data, df_hfa, by="trial")
  
  # number trials, number timepoints
  n_obs <- nrow(pt_bigdf)
  n_trials <- max(pt_bigdf$trial)
  n_times <- length(timings_w)
  
  # wins
  ga_win <- pt_gamble_data[, "win.ind"]==1
  no_win <- !ga_win
  ga_win_meanhfa <- apply(pt_elec_dat[ga_win, ], 2, mean)
  no_win_meanhfa <- apply(pt_elec_dat[no_win, ], 2, mean)
  
  # calculate empirical effective sigma:
  win_cov <- cov(pt_elec_dat[ga_win, ])
  no_win_cov <- cov(pt_elec_dat[no_win, ])
  # calculate effective sig (sqrt of average trace)
  win_sd <- sqrt(sum(diag(win_cov))/nrow(win_cov))
  no_win_sd <- sqrt(sum(diag(no_win_cov))/nrow(no_win_cov))
  eff_sig = c(win_sd, no_win_sd)
  names(eff_sig) <- c("win_sig", "nowin_sig")
  
  mean_df <- data.frame("meanhfa"=c(ga_win_meanhfa, no_win_meanhfa),
                        "win.ind"=c(rep(1, length(ga_win_meanhfa)),
                                    rep(0, length(no_win_meanhfa))),
                        "timings_w"=rep(timings_w, 2))
  return(list("pt_bigdf" = pt_bigdf,
              "pt_gamble_data" = pt_gamble_data,
              "pt_elec_dat" = pt_elec_dat,
              "n_obs" = n_obs,
              "n_trials" = n_trials,
              "n_times" = n_times,
              "win_mean"=ga_win_meanhfa,
              "nowin_mean"=no_win_meanhfa,
              "win_cov" = win_cov,
              "nowin_cov" = no_win_cov,
              "eff_sig" = eff_sig))
}

# gen_dat_Saez3(pt_gamble_data, pt_elec_dat, timings_w, n_win_nowin = c(50, 60), interv0 = c(-900, 0), ... ----
# made interv0, n_win_nowin, sig optional.  If NOT specified, uses values from data
#   - interv0 not specified:       no coerced null interval
#   - n_win_nowin not specified:   use numbers from data
#   - sig not specified:           use unscaled empirical covariance for win vs no-win
gen_dat_Saez3 <- function(pt_gamble_data, pt_elec_dat, timings_w, 
                          interv0=NULL, n_win_nowin=NULL, sig=NULL,
                          same_cov=FALSE,
                          want_plot=FALSE,
                          alph=75){
  # if interv0's range lands directly on elements of timings_w, 
  # those elements will also be made 0
  
  require(MASS) # needed for multivariate normal
  
  # mean hfa for each timepoint, across trials
  ga_win <- pt_gamble_data[, "win.ind"]==1
  no_win <- !ga_win
  
  if(is.null(n_win_nowin)) {
    n_win_nowin <- c(sum(ga_win), sum(no_win))
  }

  ga_win_meanhfa <- apply(pt_elec_dat[ga_win, ], 2, mean)
  no_win_meanhfa <- apply(pt_elec_dat[no_win, ], 2, mean)
  
  # coerce ga_win_meanhfa and no_win_meanhfa to be the same
  # for obs in the null interval
  
  if(!is.null(interv0)){
    if (is.list(interv0)){
      #multiple intervals
      interv0_inds <- NULL
      for (lind in 1:length(interv0)){
        interv0_inds <- c(
          interv0_inds, 
          intersect(
            which(timings_w >= interv0[[lind]][1]),
            which(timings_w <= interv0[[lind]][2])
            )
          )
      }
      
    } else {
      interv0_inds <- intersect(
        which(timings_w >= interv0[1]),
        which(timings_w <= interv0[2])
        )  
    }
    
    ga_win_meanhfa[interv0_inds] <- no_win_meanhfa[interv0_inds]
  }
  
  # get empirical covariance matrix for subgroups
  ga_win_cov <- cov(pt_elec_dat[ga_win, ])
  no_win_cov <- cov(pt_elec_dat[no_win, ])
  
  if (same_cov){ga_win_cov <- no_win_cov}
  
  # generate data via multivariate normal
  # each row = 1 trial, each col = 1 time point
  if(!is.null(sig)){
    # scale according to trace (so trace = nrow), then multiply by sigma
    ga_win_div <- sum(diag(ga_win_cov))/nrow(ga_win_cov)
    no_win_div <- sum(diag(no_win_cov))/nrow(no_win_cov)
    ga_win_cov <- sig^2*ga_win_cov/ga_win_div
    no_win_cov <- sig^2*no_win_cov/no_win_div
  }
  
  # calculate effective sig (sqrt of average trace)
  ga_win_sd <- sqrt(sum(diag(ga_win_cov)) / nrow(ga_win_cov))
  no_win_sd <- sqrt(sum(diag(no_win_cov)) / nrow(no_win_cov))
  eff_sig = c(ga_win_sd, no_win_sd)
  names(eff_sig) <- c("win_sig", "nowin_sig")
  
  # simulate data and make dataframe
  ga_win_simdat <- MASS::mvrnorm(
    n     = n_win_nowin[1], 
    mu    = ga_win_meanhfa, 
    Sigma = ga_win_cov
    )
  no_win_simdat <- MASS::mvrnorm(
    n     = n_win_nowin[2], 
    mu    = no_win_meanhfa, 
    Sigma = no_win_cov
    )
  
  ga_win_simdat_vec <- as.vector(t(ga_win_simdat))
  no_win_simdat_vec <- as.vector(t(no_win_simdat))
  
  simdat <- data.frame(
    "timings_w" = rep(timings_w, sum(n_win_nowin)),
    "trial"     = rep(1:sum(n_win_nowin), each=length(timings_w)),
    "win.ind"   = c(rep(1, times=n_win_nowin[1]*length(timings_w)),
                    rep(0, times=n_win_nowin[2]*length(timings_w)) ),
    "hfa"       = c(ga_win_simdat_vec,
                    no_win_simdat_vec))
  
  meandat <- data.frame(
    "hfa" = c(ga_win_meanhfa, no_win_meanhfa),
    "timings_w" = rep(timings_w, 2),
    "win.ind" = rep(c(1,0), each=length(timings_w))
    )
  
  
  p <- ggplot(simdat) +  
    geom_line(data = simdat, 
              aes(y = hfa, 
                  x = timings_w, 
                  group = trial, 
                  color = factor(win.ind)),
              alpha = 0.2) + 
    geom_line(data = meandat,
              aes(y=hfa, 
                  x=timings_w, 
                  color=factor(win.ind)),
              size=1.5) + 
    labs(title = "single patient, single electrode",
         subtitle = paste0(
           sum(n_win_nowin), " total trials: ",
           n_win_nowin[1], " win, ",
           n_win_nowin[2], " loss"))
  # plot data if wanted
  if (want_plot) print(p)

  return(list(
    "simdat"        = simdat,
    "win_meanhfa"   = ga_win_meanhfa,
    "nowin_meanhfa" = no_win_meanhfa,
    "timings_w"     = timings_w,
    "n_win_nowin"   = n_win_nowin,
    "interv0"       = interv0,
    "meandat"       = meandat,
    "win_cov"       = ga_win_cov,
    "nowin_cov"     = no_win_cov,
    "effsig"        = eff_sig,
    "p"             = p))
}


# gen_dat_Saez3.1(pt_bigdf, timings_w, n_win_nowin = c(50, 60), interv0 = c(-900, 0), ... ----
gen_dat_Saez3.1 <- function(pt_bigdf, timings_w, ndz0 = 20, ndz1 = 20, b0deg = 3, b1deg = 0,
                            interv0=NULL, n_win_nowin=NULL, sig=NULL,
                            same_cov=FALSE,
                            want_plot=FALSE,
                            vcov_centering = "by_trial",
                            alph=75){
  # vcov_centering --- "by_trial" or "by_interval" or "none"
  # if interv0's range lands directly on elements of timings_w, 
  # those elements will also be made 0
  require(MASS)
  require(dplyr)

  # mean hfa for each timepoint, across trials
  ga_win <- pt_bigdf %>% 
    filter(timings_w == min(timings_w)) %>% 
    dplyr::select(win.ind) == 1
  no_win <- !ga_win
  
  if(is.null(n_win_nowin)) {
    n_win_nowin <- c(sum(ga_win), sum(no_win))
  }

  ga_win_meanhfa <- pt_bigdf %>% 
    filter(win.ind==1) %>% 
    group_by(timings_w) %>% 
    summarize(means = mean(hfa))
  
  no_win_meanhfa <- pt_bigdf %>% 
    filter(win.ind==0) %>% 
    group_by(timings_w) %>% 
    summarize(means = mean(hfa))
  
  # coerce ga_win_meanhfa and no_win_meanhfa to be the same
  # for obs in the null interval
  if(!is.null(interv0)){
    if (is.list(interv0)){
      #multiple intervals
      interv0_inds <- NULL
      for (lind in 1:length(interv0)){
        interv0_inds <- c(
          interv0_inds, 
          intersect(
            which(timings_w >= interv0[[lind]][1]),
            which(timings_w <= interv0[[lind]][2])
            )
          )
      }
      
    } else {
      interv0_inds <- intersect(
        which(timings_w >= interv0[1]),
        which(timings_w <= interv0[2])
        )  
    }
    
    ga_win_meanhfa$means[interv0_inds] <- no_win_meanhfa$means[interv0_inds]
  }
  
  # estimate covariance --- LS with interval by trial interaction ----
  # construct 3-degree basis for main effect
  b0mat_obj <- basis_mat(
    x = pt_bigdf$timings_w,
    ndx = ndz0,
    bdeg = b0deg,
    cnames_parens = T,
    cnames_round = 1,
    want_plot = FALSE
  )
  B <- b0mat_obj$B
  
  
  # construct 0-degree basis for interaction
  b1mat_obj <- basis_mat(
    x = pt_bigdf$timings_w,
    ndx = ndz1,
    bdeg = b1deg,
    cnames_parens = T,
    cnames_round = 1,
    want_plot = FALSE
  )
  
  win_bal <- balance_x1(x1 = pt_bigdf$win.ind)
  ## multiply rows of interaction design (X_1) by win indicator for trial
  B_x1 <- sweep(b1mat_obj$B, 1, win_bal, "*")
  colnames(B_x1) <- paste0(colnames(b1mat_obj$B), "_x1")
  pt_bigdf$B <- B
  pt_bigdf$B_x1 <- B_x1
  pt_bigdf$B_x1_raw <- b1mat_obj$B
  winTF <- pt_bigdf$win.ind == 1
  
  
  if (vcov_centering == "none"){
      ga_win_cov2 <- pt_bigdf %>%
        filter(win.ind==1) %>% 
        dplyr::select(hfa, timings_w, trial) %>% 
        tidyr::spread(
          key = trial,
          value = hfa,
          -timings_w
        ) %>% 
      dplyr::select(-timings_w) %>% 
        t() %>% cov()
      
      no_win_cov2 <- pt_bigdf %>%
        filter(win.ind==0) %>% 
        dplyr::select(hfa, timings_w, trial) %>% 
        tidyr::spread(
          key = trial,
          value = hfa,
          -timings_w
        ) %>% 
      dplyr::select(-timings_w) %>% 
        t() %>% cov()
      
      all.equal(ga_win_cov2, ga_win_cov)
      all.equal(no_win_cov2, no_win_cov)
    
  } else if (vcov_centering == "by_trial"){
      resfit_w <- lm(hfa ~ -1 + B + B_x1 + factor(trial), data = pt_bigdf[winTF, ])
      resfit_l <- lm(hfa ~ -1 + B + B_x1 + factor(trial), data = pt_bigdf[!winTF, ])
      resid_w <- data.frame(
        resid = resfit_w$residuals, 
        trial = pt_bigdf$trial[winTF]
      )
      resid_l <- data.frame(
        resid = resfit_l$residuals, 
        trial = pt_bigdf$trial[!winTF]
      )
      resid_mat_w <- do.call(rbind, split(resid_w$resid, resid_w$trial))
      resid_mat_l <- do.call(rbind, split(resid_l$resid, resid_l$trial))
      
      ga_win_cov <- cov(resid_mat_w)
      no_win_cov <- cov(resid_mat_l)
      
  } else if (vcov_centering == "by_interval"){
      resfit_w <- lm(hfa ~ -1 + B + B_x1 + B_x1_raw:factor(trial), data = pt_bigdf[winTF, ])
      resfit_l <- lm(hfa ~ -1 + B + B_x1 + B_x1_raw:factor(trial), data = pt_bigdf[!winTF, ])
      resid_w <- data.frame(
        resid = resfit_w$residuals, 
        trial = pt_bigdf$trial[winTF]
      )
      resid_l <- data.frame(
        resid = resfit_l$residuals, 
        trial = pt_bigdf$trial[!winTF]
      )
      resid_mat_w <- do.call(rbind, split(resid_w$resid, resid_w$trial))
      resid_mat_l <- do.call(rbind, split(resid_l$resid, resid_l$trial))
      ga_win_cov <- cov(resid_mat_w)
      no_win_cov <- cov(resid_mat_l)
      
  } else {
      print("preprocessing method (vcov_centering) for VCov not valid")
  }

  if (same_cov){ga_win_cov <- no_win_cov}
  
  # generate data via multivariate normal
  # each row = 1 trial, each col = 1 time point
  if(!is.null(sig)){
    # scale according to trace (so trace = nrow), then multiply by sigma
    ga_win_div <- sum(diag(ga_win_cov))/nrow(ga_win_cov)
    no_win_div <- sum(diag(no_win_cov))/nrow(no_win_cov)
    ga_win_cov <- sig^2*ga_win_cov/ga_win_div
    no_win_cov <- sig^2*no_win_cov/no_win_div
  }
  
  # calculate effective sig (sqrt of average trace)
  ga_win_sd <- sqrt(sum(diag(ga_win_cov)) / nrow(ga_win_cov))
  no_win_sd <- sqrt(sum(diag(no_win_cov)) / nrow(no_win_cov))
  eff_sig = c(ga_win_sd, no_win_sd)
  names(eff_sig) <- c("win_sig", "nowin_sig")
  
  # simulate data and make dataframe
  ga_win_simdat <- MASS::mvrnorm(
    n     = n_win_nowin[1], 
    mu    = ga_win_meanhfa$means, 
    Sigma = ga_win_cov
    )
  no_win_simdat <- MASS::mvrnorm(
    n     = n_win_nowin[2], 
    mu    = no_win_meanhfa$means, 
    Sigma = no_win_cov
    )
  
  ga_win_simdat_vec <- as.vector(t(ga_win_simdat))
  no_win_simdat_vec <- as.vector(t(no_win_simdat))
  
  simdat <- data.frame(
    "timings_w" = rep(timings_w, sum(n_win_nowin)),
    "trial"     = rep(1:sum(n_win_nowin), each=length(timings_w)),
    "win.ind"   = c(rep(1, times=n_win_nowin[1]*length(timings_w)),
                    rep(0, times=n_win_nowin[2]*length(timings_w)) ),
    "hfa"       = c(ga_win_simdat_vec,
                    no_win_simdat_vec))
  
  meandat <- data.frame(
    rbind(ga_win_meanhfa, no_win_meanhfa),
    "win.ind" = rep(c(1,0), each=length(timings_w))
    )
  names(meandat) <- c("timings_w", "Ey", "win.ind")
  
  p <- ggplot(simdat) +  
    geom_line(data = simdat, 
              aes(y = hfa, 
                  x = timings_w, 
                  group = trial, 
                  color = factor(win.ind)),
              alpha = 0.2) + 
    geom_line(data = meandat,
              aes(y = Ey, 
                  x = timings_w, 
                  color = factor(win.ind)),
              size = 1.5) + 
    labs(title = "single patient, single electrode",
         subtitle = paste0(
           sum(n_win_nowin), " total trials: ",
           n_win_nowin[1], " win, ",
           n_win_nowin[2], " loss"))
  # plot data if wanted
  if (want_plot) print(p)

  return(list(
    "simdat"        = simdat,
    "win_meanhfa"   = ga_win_meanhfa,
    "nowin_meanhfa" = no_win_meanhfa,
    "timings_w"     = timings_w,
    "n_win_nowin"   = n_win_nowin,
    "interv0"       = interv0,
    "meandat"       = meandat,
    "win_cov"       = ga_win_cov,
    "nowin_cov"     = no_win_cov,
    "effsig"        = eff_sig,
    "p"             = p))
}


# plot_Saezsim(pt_num, el_num, interv0=NULL, sig=NULL, .... ----
plot_Saezsim <- function(pt_num, el_num, bmat_obj, interv0=NULL, sig=NULL,
                         n_win_nowin=NULL, seed=NULL,
                         timings_w=seq(-900, 1900, 50), 
                         ylims=NULL,
                         zoomlims=c(-3, 4),
                         path_to_Rdata){
  
  if (!missing(path_to_Rdata)){
    ptel_dat <- load_Saez(pt_num, el_num, timings_w, path_to_Rdata)
  } else {
    ptel_dat <- load_Saez(pt_num, el_num, timings_w)
  }
  
  if (missing(n_win_nowin)){
    n_win <- sum(ptel_dat$pt_gamble_data$win.ind)
    n_nowin <- nrow(ptel_dat$pt_gamble_data) - n_win
    n_win_nowin <- c(n_win, n_nowin)
  }
  
  if (!missing(seed)) set.seed(seed)
  
  simdat_obj <- gen_dat_Saez3(pt_gamble_data=ptel_dat$pt_gamble_data, 
                              pt_elec_dat=ptel_dat$pt_elec_dat, 
                              timings_w=timings_w, 
                              interv0=interv0,
                              n_win_nowin=n_win_nowin,
                              sig=sig)
  
  meandat <- data.frame("hfa"=c(simdat_obj$win_meanhfa,
                                simdat_obj$nowin_meanhfa),
                        "timings_w"=rep(timings_w, 2),
                        "win.ind"=rep(c(1,0), each=length(timings_w)))
  
  win_effsig <- round(sum(sqrt(diag(simdat_obj$win_cov)))/nrow(simdat_obj$win_cov), 2)
  nowin_effsig <- round(sum(sqrt(diag(simdat_obj$nowin_cov)))/nrow(simdat_obj$nowin_cov), 2)
  
  if(missing(sig)){
    subtitle_sig <- paste0("unscaled; effective sig: ", win_effsig, " win, ", nowin_effsig, " no_win")
  } else {
    subtitle_sig <- paste0("scaled; sig=", sig)
  }
 
  
  simdat_p <- ggplot() +
    geom_line(data=simdat_obj$simdat, aes(y=hfa, x=timings_w, group=trial, color=factor(win.ind)), alpha=.4) +
    geom_line(data=simdat_obj$meandat, aes(y=hfa, x=timings_w, color=factor(win.ind)), size=1.5) + 
    scale_x_continuous(breaks=bmat_obj$knots) + 
    # coord_cartesian(ylim=c(-3,3)) +
    labs(title=paste0("pt ", pt_num, ", elec ", el_num, 
                      ";  null (", interv0[1], ", ", interv0[2], ")"),
         subtitle=subtitle_sig,
         color="win")
  
  if (!is.null(ylims)){
    simdat_p <- simdat_p + coord_cartesian(ylim=ylims)
  }
  
  simmean_p <- ggplot() +
    # geom_line(data=simdat_obj$simdat, aes(y=hfa, x=timings_w, group=trial, color=factor(win.ind)), alpha=.1) +
    geom_line(data=simdat_obj$meandat, aes(y=hfa, x=timings_w, color=factor(win.ind)), size=1.5) + 
    coord_cartesian(ylim=zoomlims) + 
    scale_x_continuous(breaks=bmat_obj$knots) +
    labs(title=paste0("pt ", pt_num, ", elec ", el_num, 
                      ";  null (", interv0[1], ", ", interv0[2], ")"),
         subtitle=paste0("zoomed: ylim=c(", zoomlims[1], ", ", zoomlims[2], ")"),
         color="win")
  
  return(list("dat_p" = simdat_p,
              "mean_p" = simmean_p))
}




# plot_Saezdat(patient_num, elec_num, timings_w=seq(-900, 1900, 50), .... ----
plot_Saezdat <- function(patient_num, elec_num, timings_w=seq(-900, 1900, 50),
                         plots_wanted="both", # both, together, full, zoomed
                         ylims_zoomed, path_to_Rdata,
                         want_ggplots=F){
  require(ggplot2)
  require(gridExtra)

  if (!missing(path_to_Rdata)){
    Saez_datlist <- load_Saez(patient_num, elec_num, timings_w, path_to_Rdata)
  } else {
    Saez_datlist <- load_Saez(patient_num, elec_num, timings_w)
  }
  
  # store outside of list
  pt_bigdf <- Saez_datlist$pt_bigdf
  pt_gamble_data <- Saez_datlist$pt_gamble_data
  pt_elec_dat <- Saez_datlist$pt_elec_dat
  n_obs <- Saez_datlist$n_obs
  n_trials <- Saez_datlist$n_trials
  n_times <- Saez_datlist$n_times
  win_mean <- Saez_datlist$win_mean
  nowin_mean <- Saez_datlist$nowin_mean
  n_win <- sum(Saez_datlist$pt_gamble_data$win.ind)
  n_nowin <- n_trials - n_win
  
  # calculate empirical effective sigma:
  winTF <- ifelse(pt_gamble_data$win.ind==1, TRUE, FALSE)
  win_cov <- cov(pt_elec_dat[winTF, ])
  no_win_cov <- cov(pt_elec_dat[!winTF, ])
  # calculate effective sig (sqrt of average trace)
  win_sd <- sqrt(sum(diag(win_cov))/nrow(win_cov))
  no_win_sd <- sqrt(sum(diag(no_win_cov))/nrow(no_win_cov))
  eff_sig = c(win_sd, no_win_sd)
  names(eff_sig) <- c("win_sig", "nowin_sig")
  
  
  mean_df <- data.frame("meanhfa"=c(win_mean, nowin_mean),
                        "win.ind" = c(rep(1, length(timings_w)),
                                      rep(0, length(timings_w))),
                        "timings_w" = rep(timings_w, 2))
  
  base_p <- ggplot(data=pt_bigdf, 
                   aes(x=timings_w, y=hfa, col=as.factor(win.ind))) +
    geom_line(aes(group=trial), alpha=.2) +
    labs(title=paste0("obs. HFA: patient ", patient_num, 
                      ", electrode #", elec_num),
         subtitle=paste0(n_win, " wins",
                         "; ", n_nowin, " losses.",
                         "  Average sd (over time points): ", 
                         round(eff_sig[1], 2), " win; ",
                         round(eff_sig[2], 2), " loss"),
         color="win") +
    geom_line(data=mean_df, aes(x=timings_w, y=meanhfa, col=as.factor(win.ind)), size=1.25) 
  
  zoomed_p <- ggplot() +
    geom_line(data=mean_df, 
              aes(x=timings_w, y=meanhfa, col=as.factor(win.ind)), 
              size=1.25) +
    labs(title="zoomed in on y-axis only",
         subtitle="sample means by time point shown",
         color="win",
         y="")

    
  
  # set zoom level
  ylims_base <- quantile(pt_bigdf$hfa, probs=c(.01, .99))
  if (missing (ylims_zoomed)){
    ylims_zoomed <- range(win_mean, nowin_mean)
  }
  base_p <- base_p + coord_cartesian(ylim=ylims_base)
  zoomed_p <- zoomed_p + coord_cartesian(ylim=ylims_zoomed)
  
  if(want_ggplots){
    return(list("base"=base_p,
                "zoomed"=zoomed_p, 
                "pt_bigdf"=pt_bigdf,
                "eff_sig"=eff_sig))
  } else {
    # together, full, zoomed
    if (plots_wanted=="both"){
      print(base_p)
      print(zoomed_p)
    } else if (plots_wanted=="together"){
      base_p <-  base_p + theme(legend.position="none")
      zoomed_p <- zoomed_p + theme(legend.position="none")
      grid.arrange(base_p, zoomed_p, ncol=2)
    } else if (plots_wanted=="full"){
      print(base_p)
    } else if (plots_wanted=="zoomed"){
      print(zoomed_p)
    }
  }
}




# raster_Saezdat(patient_num, elec_nums, ...) ----
# plots absolute differences in means for specified electrodes (or all if not specified)
raster_Saezdat <- function(patient_num, elec_nums, timings_w = seq(-900, 1900, 50),
                              want_data=FALSE, want_gg_objs=FALSE){
  
  # load/format data
  load("ofc3_prewindowed.RData")
  
  pt_gamble_data <- gamble_data %>% filter(patient==patient_num)
  pt_gamble_data$trial <- 1:nrow(pt_gamble_data)
  pt_ecog <- buttonpressw[[patient_num]]
  
  if (missing(elec_nums)) elec_nums <- 1:dim(pt_ecog)[3]
  
  # dim(pt_ecog) #(rows=trials, cols=timings, depth=electrodes)
  
  # get windowed means, sd's by win/nowin and electrode
  win_trialsTF <- pt_gamble_data$win.ind==1
  
  app_margins <- 2
  if (length(elec_nums)>1){
    app_margins <- c(2, 3)
  }
  
  win_means    <- apply(pt_ecog[win_trialsTF, , elec_nums], app_margins, mean)
  nowin_means  <- apply(pt_ecog[!win_trialsTF, , elec_nums], app_margins, mean)
  win_sigs     <- apply(pt_ecog[win_trialsTF, , elec_nums], app_margins, sd)
  nowin_sigs   <- apply(pt_ecog[!win_trialsTF, , elec_nums], app_margins, sd)
  
  # dataframes for win/nowin
  windf <- melt(win_means)
  # colnames(windf) <- c("timings_w", "electrode", "mean")
  windf$timings_w <- rep(timings_w, length(elec_nums))
  windf$electrode <- rep(elec_nums, each=length(timings_w))
  windf$sig <- melt(win_sigs)[['value']]
  windf$win.ind <- 1  
  
  nowindf <- melt(nowin_means)
  # colnames(nowindf) <- c("timings_w", "electrode", "mean")
  nowindf$timings_w <- rep(timings_w, length(elec_nums))
  nowindf$electrode <- rep(elec_nums, each=length(timings_w))
  nowindf$sig <- melt(nowin_sigs)[['value']]
  nowindf$win.ind <- 0  
  
  #calculate absolute differences, cut to categorical
  windf$sep <- nowindf$sep <- abs(windf$value - nowindf$value)
  windf$sep_cut <- nowindf$sep_cut <- cut(windf$sep, breaks = seq(0, 3, .25))
  
  #stick 'em together & make unique trial/win variable
  pt_df <- rbind(windf, nowindf)
  pt_df$elec_win.ind <- paste0(sprintf("%02d", pt_df$electrode), 
                               ifelse(pt_df$win.ind==1, "w", "n"))
  
  sep_breaks <- levels(windf$sep_cut)[1:6*2]
  abs_diffs <- ggplot(windf) +
    geom_raster(aes(y=sprintf("%03d", electrode), 
                    x=timings_w, 
                    fill=sep_cut)) + 
    scale_fill_viridis_d(drop=FALSE, breaks=sep_breaks,
                         labels=c(paste0(seq(0, 2.5, .5), "-", seq(.5, 3, .5)))) + 
    labs(title=paste0("windowed sample mean diffs; pt ", patient_num),
         subtitle="abs(win mean - no_win min)",
         x="time(ms)",
         y="electrode #",
         fill="abs diff")
  
  sigs <- ggplot(pt_df) +
    geom_raster(aes(y=elec_win.ind, x=timings_w, fill=sig)) + 
    scale_fill_viridis_c() + 
    labs(title=paste0("std devs of windowed means; pt ", patient_num),
         subtitle="split by win/nowin",
         x="time(ms)",
         y="",
         fill="wndw sd")
  
  if (want_data | want_gg_objs){
    res <- list()
    if (want_data) res$dat <- pt_df
    if (want_gg_objs) {
      res$abs_diffs <- abs_diffs
      res$sigs <- sigs
    }
    return(res)
  } else {
    print(abs_diffs)
    print(sigs)
  }
}










# lda.variogram(id, y, x)----
lda.variogram <- function( id, y, x ){
  #
  # INPUT:  id = (nobs x 1) id vector
  #          y = (nobs x 1) response (residual) vector
  #          x = (nobs x 1) covariate (time) vector
  #
  # RETURN:  delta.y = vec( 0.5*(y_ij - y_ik)^2 )
  #          delta.x = vec( abs( x_ij - x_ik ) )
  #
  uid <- unique( id )
  m <- length( uid ) 
  delta.y <- NULL
  delta.x <- NULL
  did <- NULL 
  for( i in 1:m ){
    yi <- y[ id==uid[i] ]
    xi <- x[ id==uid[i] ]
    n <- length(yi)
    expand.j <- rep( c(1:n), n )
    expand.k <- rep( c(1:n), rep(n,n) )
    keep <- expand.j > expand.k
    if( sum(keep)>0 ){
      expand.j <- expand.j[keep]
      expand.k <- expand.k[keep]
      delta.yi <- 0.5*( yi[expand.j] - yi[expand.k] )^2
      delta.xi <- abs( xi[expand.j] - xi[expand.k] )
      didi <- rep( uid[i], length(delta.yi) )
      delta.y <- c( delta.y, delta.yi )
      delta.x <- c( delta.x, delta.xi )
      did <- c( did, didi )
    }
  }
  out <- list( id = did, delta.y = delta.y, delta.x = delta.x )
  out
}



# variog_Saez(pt_num, el_num)----
# plots variograms for Saez data, split by win/no-win
# uses lda.variogram function
variog_Saez <- function(pt_num, el_num, gamble_data, buttonpressw, timings_w=seq(-900, 1900, 50), n.knots=20,
                        point_alph=0.2, ser_size=1.5, pal=c("hotpink", "deepskyblue"), n_points=2500,
                        want_gg=FALSE){
  
  require(splines)
  require(latex2exp)
  if (missing(gamble_data) || missing(buttonpressw)) load("ofc3_prewindowed.RData")
  
  # subset data
  pt_gamble_data <- gamble_data %>% filter(patient==pt_num)
  pt_gamble_data$trial <- 1:nrow(pt_gamble_data)
  pt_ecog <- buttonpressw[[pt_num]]
  n_win <- sum(pt_gamble_data$win.ind)
  n_now <- nrow(pt_gamble_data) - n_win
  
  # subset
  pt_elec_dat <- pt_ecog[,,el_num]
  
  # formatting so data is all together
  library(reshape2)
  df_hfa <- melt(t(pt_elec_dat))
  names(df_hfa) <- c("time_index", "trial", "hfa")
  df_hfa$timings_w <- rep(timings_w, nrow(pt_gamble_data))
  pt_gamble_data$trial <- 1:nrow(pt_gamble_data)
  pt_bigdf <- inner_join(pt_gamble_data, df_hfa, by="trial")
  
  
  # variogram prep
  ### generate interior knots
  min.X <- min(pt_bigdf$timings_w)
  max.X <- max(pt_bigdf$timings_w)
  step <- (max.X - min.X)/n.knots
  knots.ud <- seq( min.X, max.X, step) 
  knots.int <- knots.ud[(1 :(length(knots.ud)-1) )] + step/2
  
  # regress out smoothed mean function
  fit_marg <- lm( pt_bigdf[, 'hfa'] ~ ns( pt_bigdf[, 'timings_w'], knots=knots.int ) +
                    pt_bigdf$win.ind*ns( pt_bigdf[, 'timings_w'], knots=knots.int ) )
  resids <- pt_bigdf[, 'hfa'] - fitted( fit_marg )
  winTF <- ifelse(pt_bigdf$win.ind==1, TRUE, FALSE)
  
  
  #variogram function
  out <- lda.variogram( id=pt_bigdf[, 'trial'], y=resids, x=pt_bigdf[, 'timings_w'] )
  dr <- out$delta.y
  dt <- out$delta.x
  
  out_win <- lda.variogram( id=pt_bigdf[winTF, 'trial'], y=resids[winTF], x=pt_bigdf[winTF, 'timings_w'] )
  dr_win <- out_win$delta.y
  dt_win <- out_win$delta.x
  
  out_now <- lda.variogram( id=pt_bigdf[!winTF, 'trial'], y=resids[!winTF], x=pt_bigdf[!winTF, 'timings_w'] )
  dr_now <- out_now$delta.y
  dt_now <- out_now$delta.x
  
  # total variance
  var.est <- var( resids )
  var.est_win <- var( resids[winTF] )
  var.est_now <- var( resids[!winTF] )
  
  
  #plot prep
  variog_df <- data.frame("gamma_u"=dr,
                          "delta_u"=dt,
                          "win"=winTF)
  variog_df_win <- data.frame("gamma_u"=dr_win,
                              "delta_u"=dt_win)
  variog_df_now <- data.frame("gamma_u"=dr_now,
                              "delta_u"=dt_now)
  gg_rows <- sample(1:dim(variog_df)[1], n_points)
  gg_rows_win <- sample(1:dim(variog_df_win)[1], n_points)
  gg_rows_now <- sample(1:dim(variog_df_now)[1], n_points)
  
  sm_sp <- smooth.spline( dt, dr, df=5)
  smdf <- data.frame("x"=sm_sp$x,
                     "y"=sm_sp$y)
  sm_sp_win <- smooth.spline( dt_win, dr_win, df=5)
  smdf_win <- data.frame("x"=sm_sp_win$x,
                         "y"=sm_sp_win$y)
  sm_sp_now <- smooth.spline( dt_now, dr_now, df=5)
  smdf_now <- data.frame("x"=sm_sp_now$x,
                         "y"=sm_sp_now$y)    
  
  ylims <- c(0,1.2*max(var.est, var.est_win, var.est_now))
  
  # plots
  variog_p <- ggplot(variog_df[gg_rows,], aes(y=gamma_u, x=delta_u)) + 
    geom_point(alpha=point_alph, color="grey") + 
    geom_hline(yintercept=var.est, color = "black") + 
    coord_cartesian(ylim=ylims) +
    geom_line(data=smdf, aes(y=y, x=x), color="black", size=1) + 
    labs(title = paste0("Variogram: pt", pt_num, 
                        " el", el_num,
                        " (", n_win, " wins, ",
                        n_now, " losses)"),
         subtitle = TeX('subset of $\\gamma(u)$ shown'),
         y = TeX('$\\gamma(u)$'),
         x = TeX('$\\Delta(u)$'))
  
  variog_p_winnow <- variog_p + 
    geom_line(data=smdf_win, aes(y=y, x=x), color=pal[2], size=1, linetype=5) +
    geom_line(data=smdf_now, aes(y=y, x=x), color=pal[1], size=1, linetype=5) + 
    geom_hline(yintercept=var.est_win, color = pal[2], linetype=5) + 
    geom_hline(yintercept=var.est_now, color =pal[1], linetype=5) +
    labs(title = paste0("Variogram: pt", pt_num, 
                        " el", el_num,
                        " (", n_win, " wins, ",
                        n_now, " losses)"),
         subtitle = "marginal in black; win in blue; loss in red",
         y = TeX('$\\gamma(u)$'),
         x = TeX('$\\Delta(u)$'))
  
  variog_p_now <- ggplot(variog_df_now[gg_rows_now,], aes(y=gamma_u, x=delta_u)) + 
    geom_point(alpha=point_alph, color="grey") + 
    geom_hline(yintercept=var.est_now, color = pal[1]) +
    coord_cartesian(ylim=ylims) +
    geom_line(data=smdf_win, aes(y=y, x=x), color=pal[1], size=1) +
    labs(title=paste0("Variogram: pt", pt_num,
                      " el", el_num,
                      ": ", n_now, " NO-WIN trials only"),
         subtitle = TeX('subset of $\\gamma(u)$ shown (no-win trials only)'),
         y = TeX('$\\gamma(u)$'),
         x = TeX('$\\Delta(u)$'))
  
  variog_p_win <- ggplot(variog_df_win[gg_rows_win,], aes(y=gamma_u, x=delta_u)) + 
    geom_point(alpha=point_alph, color="grey") + 
    geom_hline(yintercept=var.est_win, color = pal[2]) +
    coord_cartesian(ylim=ylims) +
    geom_line(data=smdf_win, aes(y=y, x=x), color=pal[2], size=1) +
    labs(title=paste0("Variogram: pt", pt_num,
                      " el", el_num,
                      ": ", n_win, " WIN trials only (win trials only)"),
         subtitle = TeX('subset of $\\gamma(u)$ shown'),
         y = TeX('$\\gamma(u)$'),
         x = TeX('$\\Delta(u)$'))
  
  
  if (want_gg){
    return(list("marg"=variog_p,
                "subbed"=variog_p_winnow,
                "win"=variog_p_win,
                "nowin"=variog_p_now))
  } else {
    print(variog_p_winnow)
  }
}

# variog_Saez(5, 29)
# vs <- variog_Saez(5, 29, want_gg = TRUE)
# vs
# 
# grid.arrange(vs$marg,
#              vs$win + labs(title="", subtitle="win trials only", y="", x=""),
#              vs$nowin + labs(title="", subtitle="no-win trials only", y="", x=""),
#              layout_matrix=rbind(c(1, 1, 2),
#                                  c(1, 1, 3)))














#### FOR SALARY DATA ----
# gen_dat_salMF(interv0, n_sim_obs, ....) ----
# generates independent observations mimicking salary data
# requires salary.Rdata in same folder as file
# only stratifies by male/female
gen_dat_salMF <- function(interv0, n_sim_obs,     # if n_sim_obs missing, uses nrow(salary)
                          salary, 
                          bin_size = 0.05,        # quantile increments of exp1
                          init_span = 0.4,        # overall desired smoothness
                          null_smoothspan = 0.2,  # span used to smooth out jump
                          alph=0.1, 
                          ylims=c(-2,2)){         # sim data plot limits
  require(dplyr)
  require(ggplot2)
  if (missing(salary)) load("salary.RData")
   
  if (missing(n_sim_obs)) n_sim_obs <- nrow(salary)
  
  #### data prep ----
  ## make bins in exp1
  sal <- salary %>% mutate(bin=cut(exp1, 
                                   quantile(exp1, seq(0, 1, bin_size)), 
                                   include.lowest=T))
  
  # for plotting
  sal_bins <- bin_gglabs <- levels(sal$bin)
  bin_gglabs[-seq(1, length(levels(sal$bin)), 3)] <- ""
  
  
  #### smooth mean functions ----
  # initial loess fits to original binned means by sex
  # control option required to make loess predict method extrapolate
  fit_m <- loess(lnw ~ exp1, span=init_span,
                 data=subset(sal, female==0),
                 control = loess.control(surface = "direct")) 
  fit_f <- loess(lnw ~ exp1, span=init_span,
                 data=subset(sal, female==1),
                 control = loess.control(surface = "direct"))
  
  # new covariates for predictions
  newX <- data.frame("exp1"=unique(sal$exp1))
  yhat_m <- predict(fit_m, newX)
  yhat_f <- predict(fit_f, newX)
  
  
  # coerce null region
  in_null <- newX$exp1 > interv0[1] & newX$exp1 < interv0[2]
  yhat_f0 <- ifelse(in_null, yhat_m, yhat_f)
  
  # uncoerced loess fits
  ggdf <-data.frame("yhat"=c(yhat_m, yhat_f),
                    "female"=rep(c(0,1), each=length(newX$exp1)),
                    "exp1"=rep(newX$exp1, 2))
  
  # coerced null loess fits; this will have a JUMP
  ggdf0 <-data.frame("yhat"=c(yhat_m, yhat_f0),
                     "female"=rep(c(0,1), each=length(newX$exp1)),
                     "exp1"=rep(newX$exp1, 2))        
  
  # plots
  loess_mean_p <- ggplot(ggdf, aes(y=yhat, x=exp1, color=factor(female))) + 
    geom_point() + 
    scale_color_discrete(direction=-1) + 
    labs(title="loess-fitted mean function, no null",
         subtitle="null region between grey lines") + 
    geom_vline(xintercept=interv0, color="darkgrey")
  
  loess_mean_p0 <- ggplot(ggdf0, aes(y=yhat, x=exp1, color=factor(female))) + 
    geom_point() + 
    scale_color_discrete(direction=-1) + 
    labs(title="loess-fitted mean function, coerced null; jumps",
         subtitle="null region between grey lines") + 
    geom_vline(xintercept=interv0, color="darkgrey") 
  
  
  
  
  #### smooth out the jump ----
  ## extend null interval and re-fitting loess
  # uses null_smoothspan to see how many obs to extend null interval
  exp1_ord <- newX$exp1[order(newX$exp1)]
  max_ind <- max(which(exp1_ord < interv0[2]))
  min_ind <- min(which(exp1_ord > interv0[1]))
  
  max_ind_ext <- max_ind + floor(length(exp1_ord) * null_smoothspan/2)+1
  min_ind_ext <- min_ind - floor(length(exp1_ord) * null_smoothspan/2)-1
  
  top_val <- ifelse(max_ind_ext <= length(exp1_ord),
                    exp1_ord[max_ind_ext],
                    interv0[2])
  low_val <- ifelse(min_ind_ext > 0,
                    exp1_ord[min_ind_ext],
                    interv0[1])
  
  
  # new, extended null interval
  interv0_ext <- c(low_val, top_val)
  in_null_ext <- newX$exp1 > interv0_ext[1] & newX$exp1 < interv0_ext[2]
  
  # coerce loess-generated mean functions from before & re-smooth
  yhat_f0_ext <- ifelse(in_null_ext, yhat_m, yhat_f)
  yhat_f0_sm <- loess(yhat_f0_ext ~ newX$exp1, span=null_smoothspan)$fitted
  yhat_f0_sm <- ifelse(in_null, yhat_m, yhat_f0_sm)  # re-coerce
  
  # put together data
  ggdf0_sm <-data.frame("yhat"=c(yhat_m, yhat_f0_sm),
                        "female"=rep(c(0,1), each=length(newX$exp1)),
                        "exp1"=rep(newX$exp1, 2))
  
  
  # plots
  loess_mean_p0_sm <- ggplot(ggdf0_sm, aes(y=yhat, x=exp1, color=factor(female))) +
    geom_point() +
    scale_color_discrete(direction=-1) +
    labs(title="loess-fitted mean function, no jump",
         subtitle="original interv0 between gray lines; extended interval dashed") +
    geom_vline(xintercept=interv0, color="darkgrey") +
    geom_vline(xintercept=interv0_ext, color="darkgrey", linetype="dashed")
  
  
  # gather mean functions together
  meanfunc_df <- data.frame("Ey"=c(yhat_m, yhat_f),
                            "Ey0_jump"=c(yhat_m, yhat_f0),
                            "Ey0_smooth"=c(yhat_m, yhat_f0_sm),
                            "female"=rep(c(0,1), each=length(newX$exp1)),
                            "exp1"=rep(newX$exp1, 2))
  # add same bins
  meanfunc_df$bin <- cut(meanfunc_df$exp1, (quantile(sal$exp1, seq(0, 1, 0.05))), include.lowest=T)
  
  
  
  #### binned variance ----
  ## generate bin means and sd's
  # add to sal
  sal <- sal %>% 
    group_by(female, bin) %>%
    mutate(sds=sd(lnw))
  
  # unique by bin and female (should only be of length 2*length(newX$exp1))
  sds <- sal %>% 
    group_by(female, bin) %>%
    summarize(sds=sd(lnw))
  
  datagen_df <- left_join(sds, meanfunc_df, by=c("female", "bin"))
  datagen_df <- datagen_df %>% 
    arrange(female, exp1)
  
  ## bootstrap n_sim_obs
  mother_dat <- left_join(sal, meanfunc_df)
  boot_inds <- sample(1:n_sim_obs, replace=T)
  simdat <- mother_dat[boot_inds,]
  
  # generate errors and add to make simulated y's
  simdat$eps <- rnorm(nrow(simdat), mean = 0, sd = simdat$sds)
  
  simdat$y <- simdat$Ey+simdat$eps
  simdat$y0_jump <- simdat$Ey0_jump+simdat$eps
  simdat$y0_smooth <- simdat$Ey0_smooth+simdat$eps
  
  simdat <- simdat %>%
    arrange(female, exp1)
  
  
  # plots
  nonull_p <- ggplot(simdat) + 
    geom_line(aes(x=exp1, y=Ey, color=factor(female)), size=1.5) + 
    geom_point(aes(x=exp1, y=y, color=factor(female)), alpha=alph) +
    labs(title="non-coerced sim data") + 
    ylim(ylims)
  
  jump_p <- ggplot(simdat) + 
    geom_line(aes(x=exp1, y=Ey0_jump, color=factor(female)), size=1.5) + 
    geom_point(aes(x=exp1, y=y0_jump, color=factor(female)), alpha=alph) +
    geom_vline(xintercept = interv0, color="black", linetype="dashed") +
    labs(title="coerced sim data (un-smoothed jump)",
         subtitle="null region between dashed lines") + 
    ylim(ylims)
  
  smooth_p <- ggplot(simdat) + 
    geom_line(aes(x=exp1, y=Ey0_smooth, color=factor(female)), size=1.5) + 
    geom_point(aes(x=exp1, y=y0_smooth, color=factor(female)), alpha=alph) +
    geom_vline(xintercept = interv0, color="black", linetype="dashed") +
    labs(title="coerced sim data (smoothed jump)",
         subtitle="null region between dashed lines") + 
    ylim(ylims)
  
  
  return(list("interv0" = interv0,
              "n_sim_obs" = n_sim_obs,
              "bin_size" = bin_size,
              "init_span" = init_span,
              "null_smoothspan" = null_smoothspan,
              "interv0_ext" = interv0_ext,
              "nonull_mean_p" = loess_mean_p,     # plot: sim truth, no null
              "jump_mean_p" = loess_mean_p0_sm,   # plot: sim truth, coerced
              "smooth_mean_p" = loess_mean_p0_sm, # plot: sim truth, coerced & smoothed
              "nonull_p" = nonull_p,              # bootst & sim data
              "jump_p" = jump_p,                  # bootst & sim data
              "smooth_p" = smooth_p,              # bootst & sim data
              "datagen_df" = datagen_df,          # df of data-gen mean fcn + sd
              "mother_dat" = mother_dat,          # orig data w/ data-gen truth
              "boot_inds" = boot_inds,            # row indices for bootstrap
              "simdat" = simdat))                 # simulated data
}

####                        #### ----
####    SPLINE functions    #### ----
####                        #### ----

# basis_mat(x, ndx, bdeg=3, xl=min(x), xr=max(x), cyclic=F, want_knots=T)----
basis_mat <- function(x, ndx, bdeg=3, xl=min(x), xr=max(x),
                       cyclic=F, want_knots=T, want_plot=T,
                       want_cnames=T, cnames_parens=F, cnames_round=1){
  x <- as.matrix(x,ncol=1)
  # as outlined in Eilers and Marx (1996)
  dx <- (xr - xl) / ndx                 # interval size between knots (but roundoff error!!)

  # construct knots; modified from Eilers & Marx to reduce roundoff error
  # t <- xl + dx * (-bdeg:(ndx+bdeg))  # knot values; original Eilers & Marx code; propagates roundoff error in dx
  t <- seq(xl, xr, len=(ndx+1))        # less roundoff error this way in constructing inner knots
  if(bdeg>0){                          # add exterior knots
    t_l <- xl + dx * (-bdeg:-1)        #roundoff error not as big a deal here (no obs)
    t_r <- xr + dx * (1:bdeg)
    t <- c(t_l, t, t_r)
  }
  dx <- c(diff(t), dx)
  DX <- (0 * x + 1) %*% dx              # this version will not propagate roundoff error as much
  
  T1 <- (0 * x + 1) %*% t               # knot values stacked to have n rows (n = length(x))
  X <- x %*% (0 * t + 1)                # x vector stacked to have n rows

  P <- (X - T1) / DX                    # distances between x value and knot j

  B <- (T1 <= X) & (X < (T1 + DX))      # which interval does X fall into? (N_{i,1})
  

  if(sum(rowSums(B!=0)>bdeg+1)!=0){
    warning("observations falling on knots may be causing ill-conditioned basis")
  }
  # sprintf("%.100f", t-lag(t))
  # sprintf("%.100f", diff(t))
  # x[which(rowSums(B!=0)>bdeg+1)]
  # 
  # typeof(T1)
  # typeof(X)
  # T1[72,] <= X[72,]
  # (X[72,] < (T1[72,] + dx))
  # 
  # sprintf("%.100f", T1[72, ])
  # sprintf("%.100f", dx)
  # 
  # 
  # X[72,]
  # T1[72, 3:4]
  # T1[72, 3:4]+dx
  # 
  # sprintf("%.100f", T1[1, ]+dx)
  # sprintf("%.100f", X[, 1])
  
  r <- c(2:length(t), 1)              # 
  
  if(bdeg>0){
    for (k in 1:bdeg){
      B <- (P * B + (k + 1 - P) * B[ ,r]) / k; 
    }
  } else if(bdeg==0){
    B <- (T1 <= X) & (X < (T1 + DX))
    # adjustment for last column
    need_adj <- (x == xr)
    if(sum(need_adj)>0){
      warning(paste0("Max x value matches last knot; observations at max x value placed in final basis. \n",
                     "  Consider specifying xr = max(x) + small delta"))
    }
    B <- apply(B, 2, function(x) ifelse(x==T, 1, 0))
    B[need_adj, ndx] <- 1
  }
  
  
  B <- B[,1:(ndx+bdeg)]
  
  if (cyclic){
    for (i in 1:bdeg){
      B[ ,i] <- B[ ,i] + B[ ,ndx+i]    
    }
    B <- B[ , 1:(ndx)]
  }
  
  
  if (sum(round(rowSums(B), 5)!=1)>0){
    warning(paste0("check basis - irregular values returned \n",
                   "  may need to widen bounds slightly (specify xl and xr)"))
  }
  
  if(want_cnames){
    closed_lb <- ifelse(cnames_parens, "(", "[")
    B_cnames <- paste0(closed_lb, round(t[1:(dim(B)[2])], cnames_round), 
                       ":", round(t[1:(dim(B)[2])+(bdeg+1)], cnames_round), 
                       ")") #column names indicate curve's active interval
    colnames(B) <- B_cnames
  }
  
  basis_supports <- cbind(t[1:(dim(B)[2])],
                          t[1:(dim(B)[2])+(bdeg+1)]
  )
  
  
  # make plot for visual check
  xplt <- x[order(x)]
  Bplt <- B[order(x), ]
  dfplt <- tidyr::gather(
    data = as.data.frame(Bplt),
    key = "active_interval",
    value = "value"
  )
  dfplt$x <- rep(xplt, times = ncol(Bplt))
  
  plt <- ggplot(dfplt) +
    geom_line(
      aes(
        y = value,
        x = x,
        color = active_interval
      )
    )
  
  
  if(want_plot){
    print(plt)
    # # visual check
    # xplt <- x[order(x)]
    # Bplt <- B[order(x), ]
    # 
    # # matplot version
    # matplot(y=Bplt, x=xplt, type='l', axes=F,
    #         main=paste0("basis: deg=", bdeg, 
    #                     ", ndx=", ndx,
    #                     ", knots at ticks"),
    #         ylab="basis projection",
    #         xlab="original var")
    # axis(1, at=t, labels=round(t, 1), las=2)
    # axis(2)
  }

  res <- list(
    plt = plt,
    B = B, 
    knots = t[(bdeg+1):(length(t)-bdeg)], 
    all_knots = t, 
    cnames = B_cnames, 
    basis_supports = basis_supports
  )
  
  return(res)
}




# # OLD VERSION _ ROUNDOFF ERRORS
# basis_mat <- function(x, ndx, bdeg=3, xl=min(x), xr=max(x), 
#                       cyclic=F, want_knots=T, 
#                       want_cnames=T, cnames_parens=F, cnames_round=1){
#   # basis matrix will have ndx+bdeg columns, each column representing a basis curve
#   x <- as.matrix(x,ncol=1)
#   
#   # as outlined in Eilers and Marx (1996)
#   dx <- (xr - xl) / ndx               # interval size between knots
#   t <- xl + dx * (-bdeg:(ndx+bdeg))   # knot values
#   T1 <- (0 * x + 1) %*% t              # knot values stacked to have n rows (n = length(x))
#   X <- x %*% (0 * t + 1)              # x vector stacked to have n rows
#   P <- (X - T1) / dx                   # distances between x value and knot j
#   B <- (T1 <= X) & (X < (T1 + dx))      # which interval does X fall into? (N_{i,1})
#   r <- c(2:length(t), 1)              # 
#   
#   if(bdeg>0){
#     for (k in 1:bdeg){
#       B <- (P * B + (k + 1 - P) * B[ ,r]) / k; 
#     }
#   } else if(bdeg==0){
#     B <- apply(B, 2, function(x) ifelse(x==T, 1, 0))
#   }
#   
#   B <- B[,1:(ndx+bdeg)]
#   
#   if (cyclic){
#     for (i in 1:bdeg){
#       B[ ,i] <- B[ ,i] + B[ ,ndx+i]    
#     }
#     B <- B[ , 1:(ndx)]
#   }
#   
#   
#   if (max(B) > 1){
#     warning(paste0("check basis - irregular values returned \n",
#                    "may need to widen bounds slightly (specify xl and xr)"))
#   }
#   
#   if(want_cnames){
#     closed_lb <- ifelse(cnames_parens, "(", "[")
#     B_cnames <- paste0(closed_lb, round(t[1:(dim(B)[2])], cnames_round), 
#                        ":", round(t[1:(dim(B)[2])+(bdeg+1)], cnames_round), 
#                        ")") #column names indicate curve's active interval
#     colnames(B) <- B_cnames
#   }
#   
#   basis_supports <- cbind(t[1:(dim(B)[2])],
#                           t[1:(dim(B)[2])+(bdeg+1)]
#                           )
#   
#   if (want_knots){
#     return(list(B=B, 
#                 knots=t[(bdeg+1):(length(t)-bdeg)], 
#                 all_knots=t, 
#                 cnames=B_cnames, 
#                 basis_supports=basis_supports))
#   } else {
#     return(B)
#   }
# }





# make_Dk(B, pen_ord=2)----
make_Dk <- function(B, pen_ord=2){
  # pen_ord = order of penalty.  
  # pen_ord=0 gives ridge regression
  # penalty influences main diagonal and K subdiagonals
  # limit is a polynomial of order k-1 if degree of B-splines is >= k
  D <- diag(ncol(B))
  for (k in 1:pen_ord) D <- diff(D)
  return(D)
}



# ps_regress(B, y, Dk, lambda)----
# frequentist p-spline regression
ps_regress <- function(B, y, Dk, lambda){
  Q <- solve(ensure_inv(mat=(t(B)%*%B + lambda*t(Dk)%*%Dk)))
  coefs <- Q%*%t(B)%*%y
  yhat <- B %*% coefs
  SSres <- sum((y-yhat)^2)
  t1 <- sum(diag(Q%*%(t(B)%*%B)))
  gcv <- SSres / ((nrow(B)-t1)^2)
  
  return(list(yhats=yhat, coefs=coefs, Q=Q, gcv=gcv, SSres=SSres, resids=y-yhat))
}


# freq_splinefit(y, B, P, lam, Q, want_vcovs=F, upper_optim=100, alpha_level=.05)----
freq_splinefit <- function(y, B, P, lam, Q, want_vcovs=F, upper_optim=100,
                           alpha_level=.05){
  # eqn 9 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3080050/
  # from Hastie & Tibshirani 1993
  # INPUT: 
  #   B = design matrix
  #   P = penalty matrix
  #   lam = lambda
  # OUPUT:
  # fitted = fitted values
  # resid = residuals
  # sigsq_hat = estiamte of variance
  # HW_vcov = Huber-White sandwich vcov estimator for coefficients
  # fitted_vcov = vcov estimator for fitted values
  # df_eff = effective degrees freedom
  
  # if lambda not specified, runs optim() to find lambda 
  # (minimizes generalized cross-validation statistic)
  if(missing(lam)){
    lam <- optim(par=.05, gcv_calc, 
                 method="Brent", 
                 lower=0, upper=upper_optim, 
                 y=y, B=B, P=P)$par
  }
  
  # Q (XtXinverse equivalent) & coefs
  n <- length(y)
  if(missing(Q)){
    Q <- solve(t(B)%*%B + n*lam*P)
  }
  coefs <- Q %*% t(B) %*% y
  
  # calculate \hat{\sigma^2}
  yhat <- B%*%coefs
  resids <- y-yhat
  SSres <- sum(resids^2)
  df_eff <- n-sum(diag(Q*(t(B)%*%B)))
  sigsq_hat <- SSres / df_eff
  
  # Huber-White vcov (robust sandwich estimator):
  HW_vcov <- sigsq_hat * (Q %*% (t(B)%*%B) %*% Q)
  if(want_vcovs){
    H <- B %*% Q %*% t(B)
    yhats_vcov <- sigsq_hat * H%*%H
  } else {
    H <- "B %*% Q %*% t(B)"
    yhats_vcov <- "sigsq_hat*(H%*%H)"
  }
  
  
  
  #gcv stat - from original ps_regress function
  gcv <- SSres / (df_eff^2)
  
  # coef results
  alph_qtile <- qt(p=1-alpha_level/2, 
                   df=df_eff)
  CI_lo <- coefs - alph_qtile*sqrt(diag(HW_vcov))
  CI_hi <- coefs + alph_qtile*sqrt(diag(HW_vcov))
  coef_mat <- cbind(coefs, CI_lo, CI_hi)
  dimnames(coef_mat)[[2]] <- c("Estimate", "CI_lo", "CI_hi")
  
  return(list("y" = y,
              "B" = B,
              "lambda" = lam,
              "Q" = Q,
              "H" = H,
              "coefs" = coefs,
              "yhat" = yhat,
              "resids" = resids,
              "SSres" = SSres,
              "df_eff" = df_eff,
              "sigsq_hat" = sigsq_hat,
              "gcv" = gcv,
              "HW_vcov" = HW_vcov,
              "yhats_vcov" = yhats_vcov,
              "coef_mat" = coef_mat))
}




# gcv_calc(y, B, P, lam)----
gcv_calc <- function(y, B, P, lam){
  Q <- solve(ensure_inv(mat=t(B)%*%B + lam*P))
  coefs <- Q%*%t(B)%*%y
  yhat <- B%*%coefs
  df_eff <- length(y)-sum(diag(Q%*%(t(B)%*%B)))
  SSres <- sum((y-yhat)^2)
  gcv <- SSres / (df_eff^2)
  return(gcv)
}





# choose_lambda(B, y, Dk, lambdas=seq(0,2, .1), want_plot=F, print_lam=F)----
choose_lambda<- function(B, y, Dk, lambdas=seq(0,2, .1), 
                         want_plot=F, print_lam=F){
  
  gcv_hold <- rep(NA, length(lambdas))
  for (i in 1:length(lambdas)){
    gcv_hold[i] <- ps_regress(B=B, y=y, Dk=Dk, lambda=lambdas[i])$gcv
  }
  best_lam <- lambdas[which(gcv_hold==min(gcv_hold))]
  
  if (print_lam) print(paste0("best lambda = ", best_lam))
  
  if (want_plot==T){
    plot(x=lambdas, y=gcv_hold, type='l', lty=2)
    points(x=lambdas, y=gcv_hold)
  }
  
  return(best_lam)
}




# logsearch_lambda(B, y, Dk, max_iter=10, verbose=F)----
logsearch_lambda <- function(B, y, Dk, max_iter=10, verbose=F){
  inc=1
  pow <- seq(-2,2,1)
  iter=0
  
  # logarithmic search loop
  while(inc>.01 && iter<=max_iter){
    iter <- iter+1
    if(verbose){paste0("iteration ", iter, 
                       "; powers: ", paste(pow, collapse=", "))}
    
    lams <- 10^pow
    lam_t <- choose_lambda(B, y, Dk, lambdas=lams, 
                           want_plot=F, print_lam=F)
    ind <- which(lams==lam_t)
    
    if (ind==1){
      pow <- pow-3
    } else if (ind==5){
      pow <- pow+3
    } else {
      newmid <- pow[ind]
      inc <- inc/2
      pow <- newmid + inc*seq(-2,2,1)
    }
  }
  
  if(verbose){paste0("END: iteration ", iter, 
                     "; powers: ", paste(pow, collapse=", "))}
  
  return(list("lam" <- lam_t, 
              "finished" <- iter<max_iter))
}









# split_basis_nonunif(x=timings_w, knots_list, bdeg=3, pen_order=2, round_dig_cnames=1)----
#makes non-uniform split basis
split_basis_nonunif <- function(x=timings_w, 
                        knots_list,
                        bdeg=3, pen_order=2,
                        round_dig_cnames=1){
  require(splines)
  check_legal <- prod(sapply(knots_list, function(x) length(x) >= bdeg+2))
  
  if (check_legal != 1) {
    cat("too few bases in splits")
    return(list("legal_split"=F))
  }
  
  all_knots_list <- lapply(knots_list, 
                           function(x) c(rep(x[1], bdeg), 
                                         x, 
                                         rep(x[length(x)], bdeg)))
  all_knots <- unlist(all_knots_list)
  basis_supports <- lapply(all_knots_list,
                           function(x) {
                             res <- cbind(x[1:(length(x)-(bdeg+1))],
                                          x[1:(length(x)-(bdeg+1))+bdeg+1])
                             colnames(res) <- c("start", "end")
                             res
                           })
  delta_intervals <- lapply(basis_supports, 
                            function(x) {
                              res <- apply(x, 2, unique)
                              colnames(res) <- c("start", "end")
                              res
                            })
  
  
  B <- NULL
  for(kn_set in knots_list){
    B_partial <- bs(x = x, 
                    knots = kn_set,
                    Boundary.knots = range(kn_set))
    #remove obs outside of range
    in_TF <- x>=range(kn_set)[1] & x <=range(kn_set)[2]
    
    #remove 0 column if exists
    B_partial[!in_TF, ] <- 0
    B_partial <- B_partial[, apply(B_partial, 2, sum)!=0]
    
    #attach basis set to design matrix
    B <- cbind(B, B_partial)
  }
  
  #add column names to B
  colnames(B) <- unlist(
    lapply(basis_supports, function(x) 
      apply(x, 1, function(y) 
        paste0(round(y[1], round_dig_cnames), 
               ": ", 
               round(y[2], round_dig_cnames)))))
  
  if((sum(B>1) + sum(B<0)) > 0 ){
    warning(paste("  basis matrix contains values above 1 or below 0",
                  "  basis may be faulty", sep="/n"))
  }
  
  return(list("legal_split"=T,
              "design"=B,
              "all_knots_list"=all_knots_list,
              "basis_supports_list"=basis_supports,
              "delta_intervals_list"=delta_intervals,
              "basis_supports"=do.call(rbind, basis_supports),
              "delta_intervals"=do.call(rbind, delta_intervals)
  ))
}


####                    #### ----
####    HELPER FUNCS    #### ----
####                    #### ----
####  FOR MATRICES  ## ----
# block_diag(mat_list)----
block_diag <- function(mat_list){
  # takes square matrices in a list
  # combines into block diagonal matrix
  matdims <- sapply(mat_list, dim)[1,]
  nr <- sum(matdims)
  resmat <- matrix(0, nrow=nr, ncol=nr)
  # loop through mat_list and place into mat
  b_end <- cumsum(matdims)
  b_st <- c(1, b_end+1)
  for(i in 1:length(mat_list)){
    resmat[  b_st[i]:b_end[i], b_st[i]:b_end[i]  ] <- mat_list[[i]]
  }
  return(resmat)
}



# check_inv(mat)----
# returns T if matrix invertible / F if not
check_inv <- function(mat) class(try(solve(mat),silent=T))=="matrix"





# ensure_inv(mat, delt=1E-5)----
ensure_inv <- function(mat, delt=1E-5){
  inv <- check_inv(mat)
  if (inv==F){
    if (dim(mat)[1]!=dim(mat)[2]){
      cat("matrix not square")
    } else {
      mat <- mat+delt*diag(dim(mat)[1])
    }
  }
  return(mat)
}



# force_solve(mat, delt=1E-5, diagnostics=F) ----
force_solve <- function(mat, delt=1E-5, diagnostics=F){
  if (dim(mat)[1]!=dim(mat)[2]){
    cat("matrix not square")
  }
  
  mat_inv <- try(solve(mat),silent=T)
  inv <- class(mat_inv)[1]=="matrix"
  
  delt_total=0
  while (inv==F){
    mat <- mat+delt*diag(dim(mat)[1])
    mat_inv <- try(solve(mat),silent=T)
    inv <- class(mat_inv)=="matrix"
    delt_total=delt_total+delt
  }
  
  if (diagnostics){
    if(delt_total!=0)   warning(paste0(delt_total, " added to diagonal to make invertible"))
  }
  return(mat_inv)
}



####  FOR MCMC  ##----

# generate gam_js from gamma vector and list structure ----
gam_js_from_gam_structure <- function(gam, gam_js_structure){
  jk <- 1
  j <- 1
  while(jk <= length(gam)){
    gam_js_structure[[j]] <- gam[jk:(jk+length(gam_js_structure[[j]])-1)]
    jk <- jk+length(gam_js_structure[[j]])
    j<- j+1
  }
  return(gam_js_structure)
}




# linspl_inds_func(gam_js)----
# find indices for gammas corresponding to linear covars and spline-modeled covars
linspl_inds_func <- function(gam_js){
  tot_lengths <- cumsum(sapply(gam_js, length))
  # gam_js_enum = gam_js filled with consecutive integers
  gam_enum <- 1:max(tot_lengths)
  gam_js_enum <- gam_js_from_gam_structure(gam=gam_enum, 
                                           gam_js_structure=gam_js)
  lin_list_inds <- which(sapply(gam_js_enum, function(x) length(x)==1)==T) #vector
  lin_inds <- unlist(gam_js_enum[lin_list_inds])
  spl_list_inds <- setdiff(1:length(gam_js), lin_list_inds)
  spl_inds <- gam_js_enum[spl_list_inds]
  if(is.list(spl_inds)==F){
    spl_inds <- list(spl_inds)
  }
  
  return(list("lin_gam_js_inds" = lin_list_inds,  #vector; use with gam_js   - as gam_js[lin_list_inds]
              "spl_gam_js_inds" = spl_list_inds,  #vector; use with gam_js   - as gam_js[spl_list_inds]
              "lin_gam_inds" = lin_inds,          #vector; use with gam
              "spl_gam_inds" = spl_inds,
              "gam_js_enum" = gam_js_enum))         #LIST; use list entries with gam
}



# intercept_needed_func <- function(gam, gam_js, B0_inds, B0_position=1)----
# NOTE:  B0_position=1
#        Assumes B_0 is the first spline-projected variable 
#        placed into gam_js (disregarding linear variables).
#        IF NOT, must specify.
intercept_needed_func <- function(gam, gam_js, 
                                  B0_inds, B0_position=1){
  # one of either linspl_inds_obj or gam_js must be provided
  # gam_js only used to find structure
  if (missing(B0_inds)) B0_inds <- linspl_inds_func(gam_js)$spl_gam_inds[B0_position]
  B0_inds <- unlist(B0_inds)
  
  ifelse( sum(gam[B0_inds])==0, 
          return(TRUE), 
          return(FALSE))
}



# des_mat_from_gam(gam, des_mat_full, intercept_needed=F)----
des_mat_from_gam <- function(gam, des_mat_full, intercept_needed=F){
  gamTF <- gam_to_TF(gam)
  des_mat <- des_mat_full[, gamTF]
  
  if(intercept_needed){
    des_mat <- cbind(1, des_mat)
    colnames(des_mat)[1] <- "intercept"
  }
  
  return(des_mat)
}





# gam_mat_func(length_gam, gam)----  
#generate matrix of all possible gammas (used for validation)
gam_mat_func <- function(length_gam, gam){
  if(missing(length_gam)){
    l_g <- length(gam)
  } else {
    l_g <- length_gam
  }
  
  gmat <- sapply(1:l_g, function(i) 
    rep(c(rep(1, 2^(l_g-i)), rep(0, 2^(l_g-i))), 2^(i-1)))
  return(gmat)
}


# gam_to_TF(gam)----
#switches 1's/0's to T/F
gam_to_TF <- function(gam){
  if(typeof(gam)=="list")  gam <- unlist(gam)
  ifelse(gam==1, T, F)
}






# num_active(gam, linspl_inds_obj) ----
# counts number active in each coef group
num_active <- function(gam, linspl_inds_obj){
  unlist(
    lapply(
      linspl_inds_obj$gam_js_enum, 
      function(X) sum(gam[X] != 0))
  )
}



####                                #### ----
####    MAKE P, Vinv                #### ----
####                                #### ----


####  P FUNCTIONS  ## ----

# make_P_j(bdeg=3, pen_order=2, ndx, num0cols=0, increm=1e-3)----
#makes penalty matrix for 1 coefficient group (1 eta_j)
make_P_j <- function(bdeg=3, pen_order=2, ndx, num0cols=0, increm=1e-3){

  id <- diag(ndx + bdeg - num0cols)
  if(pen_order==0){
    return(id)
  }
  
  Dk <- diff(id, differences=pen_order)
  return(t(Dk)%*%Dk + id*increm)
}




# make_P_full(gam_js, bdeg=3, pen_order=2)----
# generates P_full (full model's penalty matrix) from gam_js
# entries of gam_js do not matter - only used for structure
make_P_full <- function(gam_js, bdeg=3, pen_order=2){
  
  gam_j_lengths <- sapply(gam_js, length)
  gam_j_start_inds <- cumsum(c(1,gam_j_lengths))
  P_full <- diag(sum(gam_j_lengths))  #pre-populate with identity matrix
  
  for(j in 1:length(gam_j_lengths)){
    if(gam_j_lengths[j] > 1){
      # able to take vectors for bdeg, pen_ord
      bdeg_j <- ifelse(length(bdeg) == 1, 1, j)
      n_intervals <- gam_j_lengths[j]-bdeg[bdeg_j]
      P_j <- make_P_j(bdeg=bdeg[bdeg_j],
                      pen_order=pen_order[bdeg_j],
                      ndx=n_intervals,
                      num0cols=0,
                      increm=0)
      st_ind <- gam_j_start_inds[j]
      end_ind <- st_ind + gam_j_lengths[j] - 1
      P_full[st_ind:end_ind, st_ind:end_ind] <- P_j
    }
  }
  return(P_full)
}


# make_P_red(gam, P_full) ----
#gam can be either vector or gam_js (list)
make_P_red <- function(gam, P_full, intercept_needed=F){
  gamTF <- gam_to_TF(gam)
  
  # Null model case
  if(sum(gamTF)==0) return(as.matrix(1)) # for intercept only
  
  # Non-Null model
  P_red <- as.matrix(P_full[gamTF, gamTF])
  
  if(intercept_needed){
    #always placed in first column/entry
    new_dim <- dim(P_red)[1]+1
    P_int <- diag(new_dim)
    P_int[2:new_dim, 2:new_dim] <- P_red
    return(P_int)
  }
  
  return(P_red)
}



## Vinv FUNCTIONS ## ----
# V_Option1(lam, P) ----
Vinv_Option1 <- function(lam, P) (1-lam)*diag(nrow(P)) + lam*P






####                                #### ----
####    MAKE g, k, l                #### ----
####                                #### ----


####  g FUNCTIONS  ##----
# make_g----
# nothing needed here - just set g to n or n_trials


####  k FUNCTIONS  ## ----

# make_k (scaling factor to match UIP)----
# calculating k by matching traces
# k_trace_old(B, Vinv, delt=1E-5, BtBinv, n)----
k_trace_old <- function(BtBinv, Vinv, delt=1E-5, B, diagnostics=F){
  if (missing(BtBinv)){
    BtBinv <- force_solve(mat=t(B)%*%B, 
                          delt, 
                          diagnostics)
  }
  
  k <- sum(diag(BtBinv)) / sum(diag(Vinv))
  return(k)
}




####  l FUNCTIONS  ## ----
# make_l(gam, gam_js_structure, spl_gam_js_inds, want_l_js==F)----
make_l <- function(gam, spl_gam_inds, l_func=max){
  # count number of bases in each spline-projected covariate
  l_js <- unlist(lapply(spl_gam_inds, function(x) sum(gam[x]))) 
  # apply l_func and place floor at 1
  l <- ifelse(l_func(l_js) >= 1, l_func(l_js), 1)
  return(l)
}












####                     #### ----
####    PROBABILITIES    #### ----
####                     #### ----

####  PRIORS  ##----
# transition_count(vec, want_vec=F, want_inds=F)----
# counts/tracks changes in vector elements:
transition_count <- function(vec, want_vec=F, want_inds=F){
  if(length(vec)<2) return(0)
  trans_vec <- rep(1, length(vec)-1)
  for (i in 2:length(vec)){
    if (vec[i-1]==vec[i]) trans_vec[i-1] <- 0
  }
  if(want_vec) return(trans_vec)
  if(want_inds) return(which(trans_vec==1)+1)
  return(sum(trans_vec))
}




# p_gam_trans(gam_js, want_log=T, linspl_inds_obj, p_trans=0.1, diagnostics=F)----
#NOTE: gam_js must be in same form as gam_js_structure
p_gam_trans <- function(gam_js, want_log=T, linspl_inds_obj, p_trans=0.1, diagnostics=F){
  st_time <- Sys.time()
  
  if(missing(linspl_inds_obj)) linspl_inds_obj <- linspl_inds_func(gam_js)
  
  lin_gam_js_inds <- linspl_inds_obj$lin_gam_js_inds
  lin_gam_inds <- linspl_inds_obj$lin_gam_inds
  spl_gam_js_inds <- linspl_inds_obj$spl_gam_js_inds
  spl_gam_inds <- linspl_inds_obj$spl_gam_inds
  
  p <- length(lin_gam_js_inds)
  q <- length(spl_gam_js_inds)
  
  gam <- unlist(gam_js)
  lin_included <- sum(gam[lin_gam_inds])
  
  # I_js are inclusion indicators only for spline variables
  g_js_spl <- gam_js[spl_gam_js_inds]
  I_js <- unlist(lapply(spl_gam_inds, function(x) sum(gam[x])!=0))
  sgam <- lin_included + sum(I_js)
  
  # p(I_gamma) = p(I_gamma, s_gamma) = p(I_gamma|s_gamma) p(s_gamma)
  # p(s_gamma)  =  1 / (p+q+1) 
  lp_sgam <- - log(p+q+1)
  
  
  # p(I_gamma | s_gamma)  =  1/choose(p+q, s_gamma) 
  lp_Ijs_glin.sgam <- - log(choose(p+q, sgam))
  
  
  # p(gamma | I_gamma_spl, I_gamma_lin  = 
  #    prod(   p(g_j | I_j)    )   
  #    over j={1, ..., p} (spline-modeled covariates)
  
  if (sum(I_js)==0){
    # PROD(   p(g_j | I_j)    ) = 1 if all I_j's = 0
    # since if I_j = 0, then g_j = 0vec with probability 1
    lp_gspl.Ijs = 0 ## lp = log probability.  
  } else {
    gj.I1 <- g_js_spl[I_js]  ## truncating to only those gam_js that are included
    tau.I1 <- lapply(gj.I1, function(x) abs(diff(x)))
    
    # p(tau_j | I_j=1) = ptrans^sum_tau * (1-ptrans)^(length_tau - sum_tau)
    lp_tau.I1 <- unlist(
      lapply(tau.I1, function(x) 
        log(p_trans)*sum(x) + log(1-p_trans)*(length(x) - sum(x))
      )
    )
    
    # p(g_j | tau_j, I_j=1)  as vector for each g_j
    tau0vecs <- unlist(lapply(tau.I1, function(x) sum(x) == 0))
    lp_gj.tau_I1 <- log(ifelse(tau0vecs==T, 1, .5))
    
    # p(g_j | I_j=1)  as vector
    lp_gspl.Ijs = lp_tau.I1 + lp_gj.tau_I1
  }
  
  # log_prob = log p(gamma)
  log_p_gam <- lp_sgam + lp_Ijs_glin.sgam + sum(lp_gspl.Ijs)
  
  end_time <- Sys.time()
  if (want_log){
    return(log_p_gam)
  } else if (diagnostics){ 
    return(list("lp_gam" = log_p_gam,
                "lp_gamj_given_tj_Ij1" = lp_gj.tau_I1,
                "lp_tj_given_Ij1" = lp_tau.I1,
                "time" = end_time-st_time))
  } else {
    return(exp(log_p_gam))
  }
}






# p_gam_betabinom(gam_js, want_log=F, bdeg=3)----
# calculates PRIOR p(gamma) based on number of non-0 intervals / elements
p_gam_betabinom <- function(gam_js, want_log=T, linspl_inds_obj){
  # gam is a vector or list ("gam_js" in p_gam_trans function)
  if (is.list(gam_js)){
    gam_js <- unlist(gam_js)
  }
  # largest possible model size
  n_covars <- length(gam_js)
  n_included <- sum(gam_js)
  log_p_gam <- -log(n_covars+1) - log(choose(n_covars, n_included))
  
  if(want_log) return(log_p_gam)
  return(exp(log_p_gam))
}




















# lml_calc(n, Lam_o, Lam_n, y, X, mu_o, yty, XtX, a=1, b=1) ----
lml_calc <- function(n, Lam_o, y, X,
                     mu_o, yty, XtX,
                     a=1, b=1){
  
  if(missing(yty)) yty <- t(y)%*%y
  if(missing(XtX)) XtX <- t(X)%*%X
  
  a_n = a+n
  Lam_n <- XtX + Lam_o
  Lam_inv_n <- solve(Lam_n)
  b_n = b + yty + t(mu_o)%*%Lam_o%*%mu_o - t(y)%*%X%*%Lam_inv_n%*%t(X)%*%y
  
  lml <- 1/2 * (-n*log(2*pi)
                + log(det(Lam_o)) - log(det(Lam_n))
                + a*(log(b)-log(2)) - a_n*(log(b_n)-log(2))
                + lgamma(a_n/2) - lgamma(a/2)
  )
  return(as.numeric(lml))
}




# lml_from_gam(gam, n, y, des_mat_f, P_full, linspl_inds_obj ........ ----
# fixed: plus signs in lml
#        P is in the precision, not the covariance
lml_from_gam <- function(gam, n, y, des_mat_f, P_full, linspl_inds_obj,
                         yty,
                         Vinv_func=Vinv_Option1, # Vinv_option1: (1-lam)I + lam*P
                         g=length(y),
                         lam=.9, mu_prior_func=mu_0, a=1, b=1, a_n=a+length(y),
                         constant_l=FALSE, l_func=max,
                         k_func=k_trace_old,
                         linear_intercepts=TRUE, B0_inds){
  lin_int<-FALSE
  if(linear_intercepts){
    lin_int <- intercept_needed_func(gam=gam, B0_inds=B0_inds)
  }
  P0 <- make_P_red(gam=gam, P_full=P_full, intercept_needed=lin_int) # p-spline Penalty matrix
  V0 <- Vinv_func(lam=lam, P=P0) # unscaled prior precision V0 = (1-lam)I_n + lam*P0
  
  Vinv0 <- solve(V0) # unscaled prior VarCov
  B0 <- des_mat_from_gam(gam=gam, des_mat_full=des_mat_f, intercept_needed=lin_int)
  BtB0 <- t(B0)%*%B0
  BtBinv0 <- force_solve(mat=BtB0, diagnostics=FALSE)
  
  
  if (constant_l){
    l <- l_func
  } else {
    l <- make_l(gam=gam, spl_gam_inds=linspl_inds_obj$spl_gam_inds, l_func=l_func)
  }
  
  k <- k_trace_old(BtBinv=BtBinv0, Vinv=Vinv0)
  
  Lam_inv <- g*k/l * Vinv0    # Prior Var/Cov matrix (with penalty)
  Lam <- l/(k*g) * V0         # Prior precision
  Lam_n <- BtB0 + Lam         # Posterior precision
  Lam_inv_n <- solve(Lam_n)   # Posterior Var/Cov matrix
  
  ##  perform calculations for integrated likelihoods  ##
  mu0 <- mu_prior_func(gam=gam, int_needed=lin_int)
  
  
  if (sum(mu0!=0)==0){
    #if prior mean for beta = 0
    b_n <- b + yty - t(y)%*%B0%*%Lam_inv_n%*%t(B0)%*%y
  } else {
    mu_n <- Lam_inv_n %*% (Lam%*%mu0 + t(B0)%*%y)        # fixed
    b_n <- b + yty + t(mu0)%*%Lam%*%mu0 - t(mu_n)%*%Lam_n%*%mu_n
  }
  
  
  lml <- (1/2 * (-n*log(2*pi) +
                   log(det(Lam)) - log(det(Lam_n)) +
                   a*(log(b)-log(2)) - a_n*(log(b_n)-log(2))) + 
            lgamma(a_n/2) - lgamma(a/2))
  
  
  return(lml)
}


# lml_lastcalc(y, B, Lam, yty, BtB, meanzero=TRUE, mu0, a=1, b=1, n)) ----
lml_lastcalc <- function(y, B, Lam,
                          yty = NULL, BtB = NULL, 
                          meanzero=TRUE, mu0 = NULL,
                          a=1, b=1, n = NULL, premult_Sigma = NULL){
  if(is.null(yty)) yty <- t(y)%*%y
  if(is.null(BtB)) BtB <- t(B)%*%B
  if(is.null(n)) n <- length(y)
  a_n <- a+n
  Lam_n <- BtB + Lam

  if (meanzero){
    # # if prior mean for beta = 0
    # Lam_inv_n <- solve(Lam_n)
    # b_n <- b + yty - t(y)%*%B0%*%Lam_inv_n%*%t(B0)%*%y
    
    # faster than using solve(Lam_n)
    v <- t(y)%*%B
    b_n <- b + yty - t(solve(Lam_n, t(v))) %*% t(v)
  } else {
    # mu_n <- Lam_inv_n %*% (Lam%*%mu0 + t(B0)%*%y)
    # b_n <- b + yty + t(mu0)%*%Lam%*%mu0 - t(mu_n)%*%Lam_n%*%mu_n
    
    # faster than using solve(Lam_n)
    v <- t(y)%*%B + t(mu0)%*%Lam
    b_n <- b + yty + t(mu0)%*%Lam%*%mu0 - t(solve(Lam_n, t(v))) %*% t(v)
  }
  
  
  lml <- (1/2 * (-n*log(2*pi) +
                 log(det(Lam)) - log(det(Lam_n)) +
                 a*(log(b)-log(2)) - a_n*(log(b_n)-log(2))) + 
          lgamma(a_n/2) - lgamma(a/2)
          )
  
  # add Jacobian if pre-multiplying by MA(1) covariance
  if (!is.null(premult_Sigma)){
    lml <- lml - 1/2 * log(det(premult_Sigma))
  }
  return(lml)
}





# mu_0(gam, int_needed)----
#creates mean-zero vector
mu_0 <- function(gam, int_needed){
  length_mu <- sum(gam) + int_needed
  rep(0, length_mu)
}










#### PRIOR EXPECTATIONS ----
# E_zn_trans(q, lj, pi_trans)----
E_zn_trans <- function(q, lj, pi_trans){
  # prior expected zero norm of gamma
  # q = number linear terms
  # lj is a vector of number of bases in each coef group (ndz+bdeg)
  1/2*(q + sum( 
    lj*(1-pi_trans)^(lj-1) + lj/2*(1 - (1-pi_trans)^(lj-1))  ))
}



# E_zn_bb(q, lj)----
E_zn_bb <- function(q, lj){
  # prior expected zero norm of gamma
  # q = number linear terms
  # lj is a vector of number of bases in each coef group (ndz+bdeg)
  1/2*(q + sum(lj))
}


# E_sgam_trans(q,  l_j)----
E_sgam_trans <- function(q,  l_j) {
  # prior expected active coefficient GROUPS
  # q = number linear terms
  # lj is a vector of number of bases in each coef group (ndz+bdeg)
  # length(l_j) = p.  just used for uniformity of function inputs
  (length(l_j)+q)/2  
}




# E_sgam_bb(q, l_j)----
E_sgam_bb <- function(q, l_j) {
  # prior expected active coefficient GROUPS
  # q = number linear terms
  # lj is a vector of number of bases in each coef group (ndz+bdeg)
  q/2 + sum(l_j / (l_j+1))
}




# E_mgam_trans(l_j, pi_trans)----
E_mgam_trans <- function(l_j, pi_trans) {
  # prior expected number of transitions
  # lj is a vector of number of bases in each coef group (ndz+bdeg)
  pi_trans/2*sum(l_j-1)
}




# optimal_pi_trans(E_mgam, l_j)----
#should be inverse of E_mgam_trans
optimal_pi_trans <- function(E_mgam, l_j){
  # E_mgam = desired prior expected number of transitions
  # l_j = scalar/vector of number of bases in each spline projection
  #       (1 element per coefficient group/per spline-projected covariate)
  #       (ndz+bdeg)
  2*E_mgam/sum(l_j-1)
}

# #test:
# E_mgam_trans(l_j=c(8,8), pi_trans=optimal_pi_trans(2, c(8,8)))





# E_mgam_bb(l_j, pi_trans)----
E_mgam_bb <- function(l_j) {
  # prior expected number of transitions
  # lj is a vector of number of bases in each coef group (ndz+bdeg)
  sum(l_j-1)/3
}






####                           #### ----
####    MCMC chain analysis    #### ----
####                           #### ----

# PIPs_func(gam_mat, n_burn=0, round_dig=3, want_plot=T, ....     ----
PIPs_func <- function(gam_mat, n_burn=0, round_dig=3, 
                      want_plot=T, dividers, highlight_cols, maintext,
                      label_cex=1){
  n_iter <- dim(gam_mat)[1]
  PIPs <- round(apply(gam_mat[(n_burn+1):n_iter, ], 2, mean), round_dig)
  
  if(missing(maintext)){
    maintext <- paste0("PIP's, n_iter=", n_iter, "; n_burn=", n_burn)
  }
  
  if(want_plot){
    PIP_pal <- rep("black", ncol(gam_mat))
    if(missing(highlight_cols)==F){
      PIP_pal[highlight_cols] <- "hotpink"      
    }
    
    plot(y=PIPs, x=1:ncol(gam_mat), pch=20,
         xaxt='n', xlab='',
         col=PIP_pal,
         main=maintext)
    axis(1, at=1:ncol(gam_mat), labels=colnames(gam_mat), 
         las=2, cex.axis=label_cex) #labels
    
    
    if(missing(dividers)==F){
      abline(v=dividers, lty=2, col="grey")
    }
  }
  return(PIPs)
}






# check_active_deltas(gam, linspl_inds_obj, ...)----
# function to check active deltas
check_active_deltas <- function(gam, linspl_inds_obj, 
                                bdeg=3, want_vec=T,
                                delt_names){
  delt <- lapply(linspl_inds_obj$gam_js_enum, function(x) x[1:(length(x)-bdeg)])
  
  for( spl_grp in 1:length(linspl_inds_obj$spl_gam_js_inds)){
    grp_inds <- linspl_inds_obj$spl_gam_inds[[spl_grp]]
    
    for(i in 1:(length(grp_inds)-bdeg)){
      st_ind <- grp_inds[i]
      end_ind <- st_ind + bdeg
      delt[[spl_grp]][i] <- ifelse(sum(gam[st_ind:end_ind])==0, F, T)
    }
  }
  if(want_vec){
    delt <- unlist(delt)
    if(missing(delt_names)){
      return(unlist(delt)) 
    } else {
      names(delt) <- delt_names
      return(unlist(delt))      
    }
  }
  return(delt)
}






# delt_intervals(knots_all, bdeg=3)----
#extract knots_all from gen_dat obj, or all_knots from basis_mat obj
# generate delta interval names for equal-length spline bases
# (for individual spline group)
delt_intervals <- function(knots_all, bdeg=3, round_dig=2){
  delt_ints <- rep(NA, length(knots_all)-2*bdeg-1)
  for(k in (1+bdeg):(length(knots_all)-bdeg-1)){
    st <- knots_all[k]
    en <- knots_all[k+1]
    delt_ints[k-bdeg] <- paste0(round(st, round_dig), ":", round(en, round_dig))
  }
  return(delt_ints)
}






# model_size_func(gam_mat, n_burn=0, want_freq_plot=T, want_cater_plot=T) ----
model_size_func <- function(gam_mat, n_burn=0, 
                            want_freq_plot=T, want_cater_plot=T){
  n_iter <- nrow(gam_mat)
  m_sizes <- apply(gam_mat[(n_burn+1):n_iter, ], 1, sum)
  size_freq <- table(m_sizes)/sum(table(m_sizes))
  
  if(want_cater_plot){
    plot(m_sizes~c((n_burn+1):n_iter), type='l')
  }
  
  
  if(want_freq_plot){
    plot(size_freq, type='l', 
         main=paste0("model size prob, n_iter=", 
                     n_iter, "; n_burn=", n_burn))
  }
  return(m_size_tab)
}




# vis_gam(gam_mat ....----
# 
# visualize gammas as stripchart
vis_gam <- function(gam_mat, maintext="gammas selected", 
                    plot_pch="|", plot_cex=2, 
                    ylabel_cex=1, xlabel_cex=1,
                    dividers, highlight_cols,
                    add_to_existing=F, spacing=0.1, new_xlabs="new",
                    plot_colors= c("black", "hotpink")){
  n_gams <- nrow(gam_mat)
  bar_positions <- apply(gam_mat, 1, function(x) which(x==1))
  
  if (missing(highlight_cols)==F){
    row_pal <- ifelse(bar_positions[[1]] %in% highlight_cols, 
                      plot_colors[2], 
                      plot_colors[1])
  } else {
    row_pal <- rep(plot_colors[1], ncol(gam_mat))
  }
  
  if(!add_to_existing){
    #new plot
    plot(y=rep(1, length(bar_positions[[1]])), 
         x=bar_positions[[1]], 
         pch=plot_pch, 
         cex=plot_cex,
         main=maintext,
         col=row_pal,
         ylab="",
         xlab="",
         xaxt='n',
         yaxt='n',
         ylim=c(.5, n_gams+.5),
         xlim=c(1,ncol(gam_mat)))
    axis(2, at=1:ncol(gam_mat), labels=1:ncol(gam_mat), 
         las=2, cex.axis=ylabel_cex)
    axis(1, at=1:ncol(gam_mat), labels=colnames(gam_mat), 
         las=2, cex.axis=xlabel_cex)
  } else {
    # add to existing 
    bar_positions=lapply(bar_positions, function(x) x+spacing)
    axis(1, at=1:ncol(gam_mat) + spacing, labels=new_xlabs, 
         las=2, cex.axis=xlabel_cex)
    highlight_cols <- highlight_cols + spacing
  }
  
  start_gam <- ifelse(add_to_existing, 1, 2)
  for (i in start_gam:n_gams){
    if (missing(highlight_cols)==F){
      row_pal <- ifelse(bar_positions[[i]] %in% highlight_cols, 
                        plot_colors[2], 
                        plot_colors[1])
    }
    points(y=rep(i, length(bar_positions[[i]])), 
           x=bar_positions[[i]], 
           pch=plot_pch, 
           cex=plot_cex,
           col=row_pal)
  }
  
  if(missing(dividers)==F){
    abline(v=dividers, lty=2, col="grey")
  }
  
}

# # EXAMPLE of add_to_existing
# vis_gam(momfit_bb$mod_mat[1:10,], maintext= paste0("sig=", sig, ": MOMBF highest post prob (bbinom)"),
#         dividers=ncol(B)+.5, highlight_cols=null_bases, plot_cex=1.2)
# 
# vis_gam(gam_mat[head(lposts_bb_sort, 10),], maintext=paste0("sig=", sig, ": FE highest post prob (bbinom)"), 
#         dividers=ncol(B)+.5, highlight_cols=null_bases, plot_cex=1.2, 
#         add_to_existing = T, spacing=.2, new_xlabs = rep("FE",ncol(gam_mat)), xlabel_cex = .75)

# vis_gam2(gam_mat, posts, maintext, ....----
vis_gam2 <- function(gam_mat, posts, maintext, 
                     dividers, highlight_cols=NA, n_show=10, 
                     plot_pch="|", plot_cex=2, 
                     ylabel="rank", ylabel_cex=1, 
                     xlabel_cex=1,
                     margin_bltr=c(6,4,4,4),
                     add_to_existing=FALSE, spacing=0.2, new_xlabs=NA,
                     plot_colors= c("black", "hotpink"),
                     add_colors=c("grey", "pink"),
                     post_dig=4){
  
  # order
  post_ord <- order(posts, decreasing=TRUE)[1:n_show]
  gam_mat <- gam_mat[post_ord,]
  posts <- round(posts[post_ord], post_dig)
  
  yrep <- rowSums(gam_mat)
  yval <- unlist(sapply(1:length(yrep), function(X) rep(X, each=yrep[X])))
  xval <- unlist(apply(gam_mat, 1, function(X) which(X==1)))
  
  if (missing(maintext)) maintext <- paste0("best ", n_show, "models")
  
  if (!add_to_existing){
    row_pal <- ifelse(xval %in% highlight_cols, 
                      plot_colors[2], 
                      plot_colors[1])
  } else {
    row_pal <- ifelse(xval %in% highlight_cols, 
                      add_colors[2], 
                      add_colors[1])
  }
  
  if(!add_to_existing){
    #new plot
    par(mar=margin_bltr)
    plot(y=yval, 
         x=xval, 
         pch=plot_pch, 
         cex=plot_cex,
         main=maintext,
         col=row_pal,
         ylab=ylabel,
         bty='n', 
         xlab="",
         xaxt='n',
         yaxt='n',
         ylim=c(.5, length(yrep)+.5),
         xlim=c(1,ncol(gam_mat)))
    #x-axis labels
    axis(1, at=1:ncol(gam_mat), labels=colnames(gam_mat), 
         las=2, cex.axis=xlabel_cex)  
    
    if(!missing(posts)){
      # posts provided, display on y-axis
      axis(2, at=1:nrow(gam_mat), labels=posts,
           tick=FALSE, line=-1, las=2, 
           cex.axis=ylabel_cex, col.axis=plot_colors[1])
    } else {
      # posts not provided, just number
      axis(2, at=1:nrow(gam_mat), line=-.5, labels=1:nrow(gam_mat), 
           las=2, cex.axis=ylabel_cex)
    }
    
  } else {
    # add to existing 
    xval_inc <- xval + spacing
    points(y=yval, 
           x=xval_inc, 
           pch=plot_pch, 
           cex=plot_cex,
           col=row_pal)
    axis(1, at=1:ncol(gam_mat) + spacing, labels=new_xlabs, 
         las=2, cex.axis=xlabel_cex)
    
    if(!missing(posts)){
      # posts provided, display on R-H side
      axis(4, at=1:nrow(gam_mat), labels=posts,
           tick=FALSE, line=-1, las=2, 
           cex.axis=ylabel_cex, col.axis=add_colors[1])
    }
    
  }
  
  if(!missing(dividers)){
    abline(v=dividers, lty=2, col="grey")
  }
  
}

# EXAMPLE: 
# vis_gam2(gam_mat, posts=posts_tr,
#          maintext=paste0("highest post prob gammas (pen+trans); ",
#                          "sig=", sig, "\n Zellner+trans in grey"),
#          dividers=ncol(B)+.5, highlight_cols=highlight_cols, n_show=5,
#          plot_pch="|", plot_cex=2, 
#          ylabel="post prob", ylabel_cex=1, xlabel_cex=1,
#          margin_bltr=c(6,4,4,4),
#          add_to_existing=FALSE, spacing=0.2, new_xlabs=NA,
#          plot_colors= c("black", "hotpink"),
#          add_colors=c("grey", "pink"),
#          post_dig=4)
# vis_gam2(momfit_tr$mod_mat, posts=momfit_tr$posts, add_to_existing=TRUE, 
#          highlight_cols=highlight_cols, n_show=5, 
#          plot_pch="|", plot_cex=2, 
#          xlabel_cex=1, spacing=0.2, new_xlabs=NA,
#          plot_colors= c("black", "hotpink"),
#          add_colors=c("grey", "pink"),
#          post_dig=4)




# vis_gam_comp(gam_mat, pmat, maintext, pmat_cnames ...   ----
# provided matrix of post probs (pmat) from multiple formulations,
# selects top n_show models from each and makes stripchart with probabilities
# pmat = cbind post probs from multiple sources
vis_gam_comp <- function(gam_mat, pmat, maintext, pmat_cnames,
                         n_show=5, 
                         round_dig=3,
                         p_spacing=2.5,
                         mar_bltr,  #=c(8,14,2,2)
                         plot_cex=1, prob_cex=1,
                         plot_pch="|",
                         highlight_cols=NA,
                         plot_colors=c("hotpink", "black"),
                         hi_lo_colors=c("hotpink", "grey")){
  
  if(missing(pmat_cnames)) pmat_cnames <- colnames(pmat)
  if(length(pmat_cnames)==0) pmat_cnames <- NA
  if(missing(maintext)) maintext <-"top gamma comparisons"
  
  comp_n <- ncol(pmat)
  best_inds <- apply(pmat, 2, function(X) order(X, decreasing=TRUE)[1:n_show])
  best_inds <- unique(as.vector(best_inds))
  comp_mat <- round(pmat[best_inds, ], round_dig)
  
  #plot gammas
  y_n <- length(best_inds)
  yrep <- rowSums(gam_mat[best_inds, ])
  yval <- unlist(sapply(1:length(yrep), function(X) rep(X, each=yrep[X])))
  xval <- unlist(apply(gam_mat[best_inds, ], 1, function(X) which(X==1)))
  col_pal <- ifelse(xval %in% highlight_cols, 
                    plot_colors[1], 
                    plot_colors[2])
  
  if(missing(mar_bltr)){
    mar_bltr <- c(7, (comp_n+1)*p_spacing, 2, 2 )
  }
  par(mar=mar_bltr)
  plot(y=yval, 
       x=xval, 
       pch=plot_pch, 
       cex=plot_cex,
       main=maintext,
       col=col_pal,
       ylab="",
       bty='n', 
       xlab="",
       xaxt='n',
       yaxt='n',
       ylim=c(.5, length(yrep)+.5),
       xlim=c(1,ncol(gam_mat[best_inds, ])))
  #x-axis labels  
  axis(1, at=1:ncol(gam_mat[best_inds, ]), labels=colnames(gam_mat[best_inds, ]), 
       las=2, cex.axis=1)  
  
  for (j in 1:comp_n){
    textcol <- rep(hi_lo_colors[2], y_n+1)
    textcol[order(comp_mat[,j], decreasing=TRUE)[1:2]] <- hi_lo_colors[1]
    
    text(x=(p_spacing-1)-p_spacing*j, y=1:(y_n+1),  
         labels=c(comp_mat[,j], pmat_cnames[j]), 
         col=textcol, cex=prob_cex,
         xpd=TRUE)
  }
}
# hfreq_gams(gam_hold, n_hfreq) ----
hfreq_gams <- function(gam_hold, n_hfreq){
  gmat_counts <- data.frame(gam_hold) %>% group_by_all %>% count
  hfreq_to_lfreq_rows <- rev(order(gmat_counts$n))
  best_gam_rows <- head(hfreq_to_lfreq_rows, n_hfreq)
  
  best_gams_with_n <- gmat_counts[best_gam_rows,]
  
  best_gams <- as.matrix(best_gams_with_n[,1:(ncol(best_gams_with_n)-1)])
  
  return(list("gam_counts"=gmat_counts,
              "hfreq_to_lfreq_rows"=hfreq_to_lfreq_rows,
              "gam_counts_sorted"=gmat_counts[hfreq_to_lfreq_rows,],
              "best_gams"=best_gams,
              "best_gams_nvisits"=best_gams_with_n$n,
              "best_gams_freq"=best_gams_with_n$n/sum(best_gams_with_n$n)
  ))
}

































####                  #### ----
####    pi_jk_func    #### ----
####                  #### ----
# pi_jk_func ----
pi_jk_func <- function(gam_t, jk, y, n, des_mat_full, P_full, ndz, 
                       yty,    #calculate during initialization
                       lam=.9, bdeg=3, a=1, b=1, a_n=a+length(y),
                       prior_gam_func=p_gam_betabinom,
                       # if prior_gam_func=p_gam_trans, set value for p_trans in ...
                       gam_js_structure, linspl_inds_obj, 
                       B0_inds=linspl_inds_obj$spl_gam_inds[[1]],
                       linear_intercepts=T,
                       mu_prior_func,          # missing specifies 0 mean prior on coefs
                       Vinv_func=Vinv_Option1, # Vinv_option1: (1-lam)I + lam*P
                       g=length(y),            # 
                       constant_l=F,           # = T, specify value for l in l_func
                       l_func=max,             # contant_l = ndz+bdeg+1 works well too
                       BtBinv_diagnostics=T, ...){
  
  # set up gammas to compare
  gam1 <- gam0 <- gam_t
  gam1[jk] <- 1
  gam0[jk] <- 0
  
  
  #linear intercept needed?                                     
  #option to always exclude (linear_intercepts)
  lin_int0 <- lin_int1 <- FALSE
  if(linear_intercepts){
    lin_int0 <- intercept_needed_func(gam=gam0, B0_inds=B0_inds)
    lin_int1 <- intercept_needed_func(gam=gam1, B0_inds=B0_inds)
  }
  
  ##  make P  ##
  P0 <- make_P_red(gam=gam0, P_full=P_full, intercept_needed=lin_int0)
  P1 <- make_P_red(gam=gam1, P_full=P_full, intercept_needed=lin_int1)
  
  ##  make V - unscaled precision; specify Vinv formula in Vinv_func  ##
  V0 <- Vinv_func(lam=lam, P=P0)
  V1 <- Vinv_func(lam=lam, P=P1)
  
  ##  make B (reduced design matrices) and BtB  ##
  B0 <- des_mat_from_gam(gam=gam0, des_mat_full=des_mat_full, intercept_needed=lin_int0)
  B1 <- des_mat_from_gam(gam=gam1, des_mat_full=des_mat_full, intercept_needed=lin_int1)
  
  BtB0 <- t(B0)%*%B0
  BtB1 <- t(B1)%*%B1
  
  ##  make Lam  ##
  # make l
  if (constant_l){
    l0 <- l1 <- l_func
  } else {
    l0 <- make_l(gam=gam0, spl_gam_inds=linspl_inds_obj$spl_gam_inds, l_func=l_func)
    l1 <- make_l(gam=gam1, spl_gam_inds=linspl_inds_obj$spl_gam_inds, l_func=l_func)
  }
  
  k0 <- sum(diag(V0)) / sum(diag(BtB0))
  k1 <- sum(diag(V1)) / sum(diag(BtB1))
  
  Lam0 <- l0/(k0*g) * V0
  Lam1 <- l1/(k1*g) * V1
  
  if (missing(mu_prior_func)){
    lml0 <- lml_lastcalc(y=y, B=B0, Lam=Lam0, 
                         yty=yty, BtB=BtB0,
                         a=a, b=b, n=n)
    lml1 <- lml_lastcalc(y=y, B=B1, Lam=Lam1, 
                         yty=yty, BtB=BtB1,
                         a=a, b=b, n=n)
  } else {
    mu0 <- mu_prior_func(gam=gam0, int_needed=lin_int0)
    mu1 <- mu_prior_func(gam=gam1, int_needed=lin_int1)
    
    lml0 <- lml_lastcalc(y=y, B=B0, Lam=Lam0, 
                         yty=yty, BtB=BtB0,
                         meanzero=F, mu0=mu0, 
                         a=a, b=b, n=n)
    lml1 <- lml_lastcalc(y=y, B=B1, Lam=Lam1, 
                         yty=yty, BtB=BtB1,
                         meanzero=F, mu0=mu1, 
                         a=a, b=b, n=n)
  }
  
  ##  prior calculations  ##
  gam_js0 <- gam_js_from_gam_structure(gam=gam0, gam_js_structure = gam_js_structure)
  gam_js1 <- gam_js_from_gam_structure(gam=gam1, gam_js_structure = gam_js_structure)
  
  lprior_gam0 <- prior_gam_func(gam_js = gam_js0, want_log=T,
                                linspl_inds_obj=linspl_inds_obj,
                                ...)
  lprior_gam1 <- prior_gam_func(gam_js = gam_js1, want_log=T,
                                linspl_inds_obj=linspl_inds_obj,
                                ...)
  
  ##  bernoulli prob of gamma_jk=1  ##
  pi_jk <- (1 + exp(lml0-lml1 + lprior_gam0-lprior_gam1))^(-1) 
  
  return(list("pi_jk" = pi_jk,
              "gam"=gam_t,
              "gam0"=gam0,
              "gam1"=gam1,
              "lpost0" = lml0 + lprior_gam0,
              "lpost1" = lml1 + lprior_gam1,
              "lml0" = lml0,
              "lml1" = lml1,
              "lprior_gam0" = lprior_gam0,
              "lprior_gam1" = lprior_gam1,
              "des_mat1"=B1,
              "des_mat0"=B0
  ))
}













# MC3_full_mspace----
MC3_full_mspace <- function(n_iter, gam_init, gam_js_structure, linspl_inds_obj, 
                            y, n=length(y), des_mat_full, ndz,
                            lam=.9, bdeg=3, pen_order=2, a=1, b=1, a_n=a+length(y),
                            prior_gam_func=p_gam_betabinom,
                            update_every=F, every=100,
                            print_finish=T,
                            # if prior_gam_func=p_gam_trans, set value for p_trans in ..
                            B0_inds=linspl_inds_obj$spl_gam_inds[[1]],
                            linear_intercepts=T,
                            mu_prior_func=mu_0,     # mu_0 specifies 0 mean prior on coefs
                            Vinv_func=Vinv_Option1, # Vinv_option1: (1-lam)I + lam*P
                            g=length(y),            # 
                            constant_l=F,           # = T, specify value for l in l_func
                            l_func=max,             # constant_l = ndz+bdeg+1 works well too
                            k_func=k_trace_old,
                            BtBinv_diagnostics=F, 
                            partial_save=F,
                            partial_save_every=1000,
                            partial_save_filename="MC3_full_mspace_partial_save",...){
  
  tot_time_st <- Sys.time()
  
  # pre-calculations
  yty <- t(y)%*%y
  
  # block diagonal with I_q and spline penalty matrices.  Order depends on gam_js_structure
  P_full <- make_P_full(gam_js=gam_js_structure, 
                        bdeg=bdeg, 
                        pen_order=pen_order)
  
  # Vinv_Option1 = (1-lambda)*I_q + lambda*P  =  prior covariance matrix       # for UIP, make analogous to n*(XtX)^-1
  Vinv_full <- Vinv_func(lam, P_full)
  
  # initialize:
  gam_t <- gam <- gam_init
  gam_hold <- pj_hold <- matrix(NA, nrow=n_iter, ncol=length(gam_init))
  lpost_hold <- rep(NA, length=n_iter)
  colnames(gam_hold) <- colnames(pj_hold) <- names(gam)
  
  tot_time <- Sys.time()
  
  # loop
  for(iter in 1:n_iter){
    
    gam_t <- gam
    update_order <- sample(1:length(gam_init), length(gam_init), replace=F)
    
    # choose and update coordinate jk
    for(jk in update_order){
      
      pi_jk_t_list <- pi_jk_func(gam_t=gam_t, jk=jk, y=y, n=n, des_mat_full=des_mat_full, P_full=P_full, ndz=ndz, 
                                 yty=yty,
                                 lam=lam, bdeg=bdeg, a=a, b=b, a_n=a_n,
                                 prior_gam_func=prior_gam_func,
                                 gam_js_structure=gam_js_structure, linspl_inds_obj=linspl_inds_obj, 
                                 B0_inds=B0_inds,
                                 linear_intercepts=linear_intercepts,
                                 mu_prior_func=mu_prior_func,
                                 Vinv_func=Vinv_func, 
                                 g=g,            
                                 constant_l=constant_l,           
                                 l_func=l_func,             
                                 k_func=k_func,
                                 BtBinv_diagnostics=BtBinv_diagnostics, ...)
      
      #draw gam_k^t
      gam_t[jk] <- rbinom(1, 1, prob=pi_jk_t_list$pi_jk)
      
      # store probability
      pj_hold[iter, jk] <- pi_jk_t_list$pi_jk
    }
    
    gam_hold[iter, ] <- gam <- gam_t
    lpost_hold[iter] <- ifelse(gam_t[jk]==1, pi_jk_t_list$lpost1, pi_jk_t_list$lpost0)
    
    # track time & save partial results
    if(update_every==T){
      st_time <- loop_time(loopindex=iter, st_time, every=every)
    }
    if(partial_save==T && iter!=1 && iter%%partial_save_every==0){
      partial_res <- list("iter"=iter, 
                          "time"=Sys.time()-tot_time_st,
                          "y"=y,
                          "des_mat_full"=des_mat_full,
                          "gam_hold_partial" = gam_hold[1:iter, ],
                          "lpost_hold_partial" = lpost_hold[1:iter],
                          "pj_hold_partial" = pj_hold[1:iter, ]
      )
      save(partial_res, file=partial_save_filename)
    }
  }
  MCMC_time <- Sys.time()-tot_time
  if (print_finish){
    print(paste0(iter, " iterations took ", MCMC_time))
  }
  
  return(list("iter"=n_iter,
              "MCMC_time"=MCMC_time,
              "y"=y,
              "des_mat_full"=des_mat_full,
              "gam_hold" = gam_hold,
              "lpost_hold" = lpost_hold,
              "pj_hold" = pj_hold))
}




























####               #### ----
####    REGRESS    #### ----
####               #### ----


# orthogonalize(xj, Sj) ----
# use to orgonalize Sj, the spline basis for xj, in relation to xj
orthogonalize <- function(xj, Sj){
  X <- cbind(1, xj)
  Q_X <- solve(t(X)%*%X)
  H_X <- X%*%Q%*%t(X)
  Sj_orthog <- (diag(dim(H_X)[1])-H_X)%*%Sj
  
  return(Sj_orthog)
}



# orthog_bases(B0, B1_raw) ----

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



# des_mat_from_gam_regress(gam, des_mat_full, intercept_needed=F, linspl_inds_obj, regress_matchvec)----
des_mat_from_gam_regress <- function(gam, des_mat_full, intercept_needed=F, linspl_inds_obj, regress_matchvec){
  
  # identify active variables  
  gam_matchvec <- ifelse(gam==1, regress_matchvec, NA)  #uses NA in case regress_matchvec contains characters
  
  # orthogonalize Sj (spline basis for xj) w.r.t. xj
  # do for each j (requires matching)
  des_mat_full_orthog <- des_mat_full
  for(i in linspl_inds_obj$lin_gam_inds){
    if (is.na(gam_matchvec[i])==F){
      spl_match_inds <- which(gam_matchvec==gam_matchvec[i])
      spl_match_inds <- setdiff(spl_match_inds, gam_matchvec[i])
      xj <- des_mat_full[, gam_matchvec[i]]
      Sj <- des_mat_full[, spl_match_inds]
      Sj_orthog <- orthogonalize(xj, Sj)
      des_mat_full_orthog[, spl_match_inds] <- Sj_orthog
    }
  }
  
  des_mat <- des_mat_from_gam(gam, des_mat_full=des_mat_full_orthog, intercept_needed=intercept_needed)
  # return(list("des_mat"=des_mat,
  # "des_mat_full_orthog"=des_mat_full_orthog))    # for testing
  return(des_mat)
}

#### TESTING
# regress_matchvec <- c(1, 2, 3, 
#                       rep(1, length(linspl_inds_obj$spl_gam_inds[[1]])),
#                       rep(3, length(linspl_inds_obj$spl_gam_inds[[2]])))
# 
# d_o <- des_mat_from_gam_regress(gam, des_mat_full, F, linspl_inds_obj, regress_matchvec)
# 
# unmatch=rep(NA, ncol(des_mat_full))
# for (i in 1:ncol(des_mat_full)){
#   unmatch[i] <- sum(d_o$des_mat_full_orthog[,i] != des_mat_full[, i])
# }
# unmatch
# ncol(d_o$des_mat)
# sum(gam)









####                           #### ----
####    REDUCED MODEL SPACE    #### ----
####                           #### ----


















####                         #### ----
####    INTERVAL MATCHING    #### ----
####                         #### ----

# intervs(xl=-900, xr=1900, ndz)----
# intervs <- function(xl=-900, xr=1900, ndz){
#   # returns intervals 
#   inc <- (xr - xl)/ndz
#   seps <- xl + inc*(0:ndz)
#   return(list("inc"=inc, "seps"=seps))
# }

intervs <- function(xl=-900, xr=1900, ndz, 
                    want_exterior=F, want_nonunif=F, bdeg=3){
  # returns intervals 
  inc <- (xr - xl)/ndz
  seps <- xl + inc*(0:ndz)
  
  if(bdeg>0 & want_exterior){
    # if bdeg=0, no exterior knots
    seps_l <- seps_r <- NULL  
    
    if(want_nonunif){
      seps_l <- rep(xl, bdeg)
      seps_r <- rep(xr, bdeg)
    } else {
      seps_l <- xl - inc*(bdeg:1)
      seps_r <- xr + inc*(1:bdeg)
    }
    
    seps <- c(seps_l, seps, seps_r)
  } 
  
  return(list("inc"=inc, "seps"=seps))
}

# intervs(xl=-900, xr=2100, ndz=10)
# intervs(xl=-1000, xr=2000, ndz=15)
# intervs(xl=-1000, xr=2000, ndz=5)



# adjust_xlxr(target=0, xl=-900, xr=1900, ndz=5, want_centered=F)----
adjust_xlxr <- function(target=0, xl=-900, xr=1900, ndz=5, want_centered=F){
  #new version fixed
  # gives smallest adjustments to range in order to hit target with interval boundary
  # while containing original range within "interior" knots   
  # QUESTION: what happens if we lift the previous line's restriction?
  
  
  # see if hits target already
  inc <- (xr - xl)/ndz
  seps <- xl + inc*(0:ndz)
  hit <- is.element(target, seps)
  sep_centers <- seps[1:ndz] + inc/2
  hit_center <- is.element(target, sep_centers)
  
  if(hit && !want_centered){
    cat(paste0("range (", xl, ", ", xr, ") has knot at ", target, "\n"))
  } else if(hit_center && want_centered){
    cat(paste0(target, " is in center of interval for range (", xl, ", ", xr, ") \n"))
  } else if(!hit && !want_centered){
    cat(paste0("range (", xl, ", ", xr, ") does not have knot at target ", target, "\n"))
    inc <- (xr - xl)/(ndz-1)
    seps <- xl + inc*(0:(ndz-1))
    min_adjustment <- min(abs(seps-target))
    closest_knot <- seps[which(abs(seps-target) == min_adjustment)]
    
    if(length(closest_knot)>1){
      closest_knot <- closest_knot[2] #if two are close, the larger one will be > target
    }
    
    if(closest_knot>target){
      xl <- xl-min_adjustment
      xr <- xr-min_adjustment + inc
    } else {
      xl <- xl+min_adjustment - inc
      xr <- xr+min_adjustment
    }
    
    inc <- (xr - xl)/(ndz)
    seps <- xl + inc*(0:ndz)
    cat(paste0("new range (", xl, ", ", xr, ") suggested to place knot at target value \n"))
  } else if (!hit_center && want_centered){
    cat(paste0(target, " not centered in an interval for range (", xl, ", ", xr, ") \n"))
    inc <- (xr - xl)/(ndz-1)
    seps <- xl + inc*(0:(ndz-1))
    sep_centers <- seps + inc/2
    min_adjustment <- min(abs(sep_centers-target))
    closest_knot_ind <- which(abs(sep_centers-target) == min_adjustment)
    closest_center <- sep_centers[closest_knot_ind]
    
    
    if(length(closest_center)>1){
      closest_center <- closest_center[2]
      # cat('two points equidistant') - doesn't matter which side is taken
    }
    
    if(closest_center>target){
      xl <- xl-min_adjustment
      xr <- xr-min_adjustment + inc
    } else {
      xl <- xl+min_adjustment - inc
      xr <- xr+min_adjustment
    }
    
    inc <- (xr - xl)/(ndz)
    seps <- xl + inc*(0:ndz)
    cat(paste0("new range (", xl, ", ", xr, ") suggested to center interval at target value \n"))
  }
  
  return(list("range"=c(xl, xr),
              "inc"=inc,
              "seps"=seps))
}





####                         #### ----
####    ID Bases, cuts       #### ----
####                         #### ----


# ID_bases_in_interval(interv, basis_supports)----
ID_bases_in_interval <- function(interv, basis_supports, tol=5){
  # identify bases with support strictly inside ONE interval
  # returns column indices corresponding to basis_supports  
  interv <- interv[order(interv)]
  
  above_lo <- which(round(basis_supports[, 1], tol) >= round(interv[1]), tol)
  below_hi <- which(round(basis_supports[, 2], tol) <= round(interv[2]), tol)
  
  res <- intersect(above_lo, below_hi)
  
  if(length(res)==0) {
    warning("no bases strictly inside interval")
    return(NULL)
  }
  return(res)
}


# ID_bases_in_interval_list(interv, basis_supports, no_nulls = TRUE)----
ID_bases_in_interval_list <- function(interv, basis_supports, no_nulls = TRUE){
  bases_list <- lapply(
    interv,
    function(X) ID_bases_in_interval(
      interv = X,
      basis_supports = basis_supports
    )
  )
  
  if(no_nulls){
    bases_list <- bases_list[lengths(bases_list)!= 0]
  }
  
  return(bases_list)
}

# ID_nullbases_inds(interv0, z, basis_supports, des_mat_f ....----
ID_nullbases_inds <- function(interv0, z, basis_supports, des_mat_f,
                              want_design_inds=T, design_str="x1",
                              tol=5){
  ## identifies bases containing OBSERVATIONS inside null interval
  ## want_design_inds=T:  returns column indices corresponding to design matrix
  ##                      otherwise returns start/end of bases
  
  if(is.null(interv0) | length(interv0)<2) return(NULL)
  # in case interv0al is given as (upper, lower)
  interv0 <- interv0[order(interv0)] 
  
  # find all observed z values outside interv0al
  z_below_lo <- z<interv0[1]
  z_above_hi <- z>interv0[2]
  
  z_outTF <- z_below_lo | z_above_hi
  z_out <- unique(z[z_outTF])
  
  # find all bases containing these values
  contain_mat <- apply(basis_supports, 1, function(X)
    z_out > X[1] & z_out< X[2])
  # find indices of bases that do not contain values in z_out
  res <- which(colSums(contain_mat)==0)
  
  if(length(res)==0) {
    warning("no bases strictly inside interv0al")
    return(NULL)
  }
  
  if(want_design_inds) {
    require(stringr)
    # identify covariate projection by matching design_str
    ScolsTF <- str_detect(colnames(des_mat_f), design_str)
    
    des_contain <- matrix(NA, nrow=length(res), ncol=ncol(des_mat_f))
    for (i in 1:length(res)){
      des_contain[i,] <- str_detect(colnames(des_mat_f), paste0(".", round(basis_supports[res[i], 1]), ".", round(basis_supports[res[i], 2]), "."))
    }
    basis_cols <- apply(des_contain, 2, function(X) sum(X)!=0)
    res <- which(ScolsTF==TRUE & basis_cols==TRUE)
    
    return(res)
  } else {
    return(res)
  }
}
# ID_bases_covering_zval(zval, basis_supports)----
ID_bases_covering_zval <- function(zval, basis_supports, tol=5){
  # identify bases that have non-0 support on a value of the projected covariate z
  # (these need to be truncated)
  # returns column indices corresponding to (rows in) basis_supports
  st_lower <- round(basis_supports[,1], tol) < round(zval, tol)
  end_higher <- round(basis_supports[,2], tol) > round(zval, tol)
  return(which(st_lower & end_higher==T))
}



# ID_knots_around_zval(zval, basis_supports, tol=5)----
ID_knots_around_zval <- function(zval, basis_supports, tol=5){
  #returns knot values
  #always returns 2 values, even when zval is on a knot
  kn_vec <- unique(as.vector(basis_supports))
  kn_vec <- kn_vec[order(kn_vec)]
  zval_on_knot <- round(zval, tol)==round(kn_vec, tol)
  if (sum(zval_on_knot)>0){
    res <- rep(kn_vec[zval_on_knot],2)
  } else {
    lo_ind <- max(which(c(zval > kn_vec) ==T))
    res <-kn_vec[lo_ind:(lo_ind+1)] 
  }
  
  return(res)
}








## cut_basis(z_cut, z, des_mat, basis_supports, verbose=T, tol=5, append_suffix = T) ----
cut_basis <- function(z_cut, z, des_mat, basis_supports, 
                      verbose=T, tol=5, append_suffix = T){
  # splits basis splines covering point z_cut into two at ending/beginning at z_cut
  # z_cut:              target cut point
  # z:                  covariate that is projected onto spline basis
  # des_mat:            design matrix
  # basis_supports:     K-by-2 matrix of interval start/end - given in output from basis_mat()
  # verbose:            if TRUE, specifies bases to be cut AND plots cut bases
  
  
  des_cols_to_cut <- ID_bases_covering_zval(zval=z_cut, basis_supports = basis_supports, tol=tol)
  
  # extract des_mat columns
  dup_lo <- dup_hi <- des_mat[ ,des_cols_to_cut]
  cut_rows_lo <- which(z >= z_cut)
  cut_rows_hi <- which(z < z_cut)
  
  dup_lo[cut_rows_lo, ] <- 0
  dup_hi[cut_rows_hi, ] <- 0
  
  if (append_suffix){
    colnames(dup_lo) <- paste0(colnames(des_mat)[des_cols_to_cut], "_lo")
    colnames(dup_hi) <- paste0(colnames(des_mat)[des_cols_to_cut], "_hi")
  }
  
  basis_supports_lo <- cbind(basis_supports[des_cols_to_cut, 1], 
                             z_cut)
  basis_supports_hi <- cbind(basis_supports[des_cols_to_cut, 1], 
                             z_cut)
  
  if(verbose){
    cat("Cutting des_mat columns at z=", z_cut, ": \n")
    print(colnames(des_mat)[des_cols_to_cut]) 
    cat("corresponding to basis_supports \n")
    print(basis_supports[des_cols_to_cut,]) 
    
    
    #plot
    dup_lo_plot <- ifelse(dup_lo==0, NA, dup_lo)
    dup_hi_plot <- ifelse(dup_hi==0, NA, dup_hi)
    dup_pal <- c(rep("deepskyblue", ncol(dup_lo)),
                 rep("hotpink", ncol(dup_lo)))
    matplot(y=cbind(dup_lo_plot, dup_hi_plot), 
            x=z, 
            col=dup_pal, 
            type="l",
            main = paste0("bases cut at z_cut=", z_cut))
    matlines(y=c(0,1), 
             x=c(z_cut, z_cut), 
             lty=2)
  }
  
  return(list("z" = z,
              "z_cut" = z_cut,
              "des_mat" = des_mat,
              "basis_supports" = basis_supports,
              "basis_supports_lo" = basis_supports_lo,
              "basis_supports_hi" = basis_supports_hi,
              "des_cols_to_cut" = des_cols_to_cut,
              "cut_lo" = dup_lo,
              "cut_hi" = dup_hi))
}

####                         #### ----
####    Testing funcs        #### ----
####                         #### ----

# gam_fullenum(length_gam, gam)----
#generate matrix of all possible gammas (no restrictions yet)
#length_gam optional
gam_fullenum <- function(length_gam, gam){
  if(missing(length_gam)){
    l_g <- length(gam)
  } else {
    l_g <- length_gam
  }
  gmat <- sapply(1:l_g, function(i)
    rep(c(rep(1, 2^(l_g-i)), rep(0, 2^(l_g-i))), 2^(i-1)))
  return(gmat)
}






















# vis_gamfits(gam_mat, pr_weights, simdat, gam_js_structure....----
vis_gamfits <- function(gam_mat, pr_weights, simdat, gam_js_structure, 
                        des_mat_f, ndz, bdeg=3, pen_ord=2, 
                        maintext, num_show=10, 
                        method="posterior", lam, g, linspl_inds_obj, l_func=max,
                        kl=FALSE, XtX_orig,
                        want_data=FALSE, want_plots=TRUE, 
                        size_by_weights=FALSE, ylims,
                        simdat_obj, timings_w,
                        return_p=FALSE){
  #methods: penlike (penalized likelihood), mle (lambda=0), posterior (lam = .9, P=), Zellner ((lam-1))
  # supplying pr_weights is optional
  if (!missing(pr_weights)){
    pr_weights_ord <- order(pr_weights, decreasing=TRUE)
    gams_show <- matrix(gam_mat[pr_weights_ord[1:num_show], ], nrow=num_show)
  } else {
    gams_show <- matrix(gam_mat[1:num_show, ], nrow=num_show)
  }
  
  if(missing(linspl_inds_obj)){
    linspl_inds_obj <- linspl_inds_func(gam_js=gam_js_structure)
  }
  
  # generate gam_mat fits
  y <- simdat$hfa
  z <- simdat$timings_w
  z_show <- seq(min(z), max(z), length.out=5*ndz)
  
  bmat_obj <- basis_mat(z_show, ndz, cnames_parens=F, bdeg=bdeg, want_plot=F)
  B <- bmat_obj$B
  B0 <- matrix(0, ncol=ncol(B), nrow=nrow(B))
  colnames(B0) <- paste0("x1_", colnames(B))
  
  des_mat1 <- cbind(B, B)
  des_mat0 <- cbind(B, B0)
  
  P_f <- make_P_full(gam_js=gam_js_structure, 
                     bdeg=bdeg, 
                     pen_order=pen_ord)
  
  #iterate through gam_mat
  yhat_mat <- matrix(NA, nrow=2*length(z_show), ncol=num_show)
  coef_mat <- matrix(NA, ncol=ncol(gam_mat), nrow=num_show)
  colnames(coef_mat) <- colnames(gam_mat)
  params_mat <- matrix(NA, ncol=7, nrow=num_show)
  colnames(params_mat) <- c("lambda", "g", "l", "k", "df_eff", "gcv", "SSres")
  
  for(i in 1:nrow(gams_show)){
    gamTF <- gam_to_TF(gams_show[i,])
    int_needed <- intercept_needed_func(gam=gams_show[i,], gam_js=gam_js_structure)
    B_r <- des_mat_from_gam(gams_show[i,], des_mat_full = des_mat_f, intercept_needed = int_needed)
    P_r <- make_P_red(gam=gams_show[i,], P_f, 
                      intercept_needed=int_needed)
    
    if (method=="penlike"){
      # spline penalty matrix (frquentist version)
      if(missing(lam)){
        #conducts search for optimal (by gcv) lambda
        freq_fit <- freq_splinefit(y=y, B=B_r, P=P_r)
      } else {
        freq_fit <- freq_splinefit(y=y, B=B_r, P=P_r, lam=lam)
      }
      
    } else if (method=="mle"){
      # unpenalized likelihood
      freq_fit <- freq_splinefit(y=y, B=B_r, P=P_r, lam=0)
      
    } else if (method=="posterior") {
      # posterior mean (mimics current formualtion)
      V_r <- Vinv_Option1(lam=.9, P=P_r)
      Vinv_r <- solve(V_r)
      BtB_r <- t(B_r)%*%B_r
      
      if(typeof(l_func)=="character"){
        if(l_func=="xtx") kl=TRUE
      }
      
      if (!kl) {
        BtBinv_r <- force_solve(mat=BtB_r, diagnostics=FALSE)
        k <- k_trace_old(BtBinv=BtBinv_r, Vinv=Vinv_r)
        l <- make_l(gam=gams_show[i,], spl_gam_inds=linspl_inds_obj$spl_gam_inds, l_func=l_func)
      } else {
        l <- 1
        if (missing(XtX_orig)){
          x1 <- simdat$win.ind
          X <- model.matrix(~z + x1*z)
          XtX_orig <- t(X)%*%X
        }
        k <- k_trace_old(BtBinv = XtX_orig, Vinv=Vinv_r)
      }
      # Lam_inv <- g*k/l * Vinv_r #penalty (covariance, not prevision) matrix
      Lam <- l/(g*k) * V_r
      Lam_n <- BtB_r + Lam         # Posterior precision
      Lam_inv_n <- solve(Lam_n)   # Posterior Var/Cov matrix
      freq_fit <- freq_splinefit(y=y, B=B_r, Q=Lam_inv_n, lam=1)
      params_mat[i, c("g", "l", "k")] <- c(g, l, k)
      
    } else if (method=="Zellner"){
      BtB_r <- t(B_r)%*%B_r
      Q <- length(y)/(length(y)+1) * solve(BtB_r)
      freq_fit <- freq_splinefit(y=y, B=B_r, Q=Q, lam=1)
    }
    
    params_mat[i, c("lambda", "df_eff", "gcv", "SSres")] <- unlist(freq_fit[c("lambda", "df_eff", "gcv", "SSres")])
    
    coef_mat[i, gamTF] <- ifelse(int_needed, freq_fit$coefs[-1], freq_fit$coefs)
    
    # make preds
    des_mat1_r <- des_mat_from_gam(gams_show[i,], des_mat_full = des_mat1, intercept_needed = int_needed)
    des_mat0_r <- des_mat_from_gam(gams_show[i,], des_mat_full = des_mat0, intercept_needed = int_needed)
    
    yhat1 <- des_mat1_r%*%freq_fit$coefs
    yhat0 <- des_mat0_r%*%freq_fit$coefs
    yhat_mat[,i] <- c(yhat1, yhat0)
  }
  #reshape for ggplot
  yhat_df <- reshape2::melt(yhat_mat)
  names(yhat_df) <- c("row", "fit_num", "yhat")
  yhat_df$win.ind <- rep(c(1,0), each=length(z_show))
  yhat_df$z <- rep(z_show, 2*num_show)
  
  # plot data
  if(want_plots){
    base_p <- ggplot() + 
      geom_line(data=simdat, aes(x=timings_w, y=hfa, group=trial + win.ind*length(hfa)), 
                color="grey")
    
    if(!missing(maintext)){
      base_p <- base_p +
        labs(title=maintext)
    }
    
    #plot gam_mat fits
    if(!missing(pr_weights) & size_by_weights){
      yhat_df$weight <- rep(pr_weights[pr_weights_ord][1:num_show], each=2*length(z_show))
      p <- base_p + 
        geom_line(data=yhat_df, 
                  aes(x=z, 
                      y=yhat, 
                      group=fit_num+win.ind*num_show, 
                      color=weight,
                      size=weight)) + 
        scale_color_viridis_c(trans="log10", direction=-1)
    } else if (!missing(pr_weights) & !size_by_weights){
      yhat_df$weight <- rep(pr_weights[pr_weights_ord][1:num_show], each=2*length(z_show))
      p <- base_p + 
        geom_line(data=yhat_df, 
                  aes(x=z, 
                      y=yhat, 
                      group=fit_num+win.ind*num_show, 
                      color=weight), size=1.5, alpha=.25) + 
        scale_color_viridis_c(trans="log10", direction=-1)
    } else {
      p <- base_p + 
        geom_line(data=yhat_df, 
                  aes(x=z, 
                      y=yhat, 
                      group=fit_num+win.ind*num_show, 
                      color=factor(win.ind)))
    }
    
    # set y limits
    if(!missing(ylims)){
      p <- p+ coord_cartesian(ylim=ylims)
    } else {
      p <- p+ coord_cartesian(ylim=quantile(y, probs=c(.025, .975)))
    }
    
    if(!missing(simdat_obj)){
      # add true means
      if(missing(timings_w)) timings_w <- seq(-900, 1900, 50)
      sim_tmean <- data.frame("timings_w"=rep(timings_w, 2),
                              "hfa"=c(simdat_obj$win_meanhfa,
                                      simdat_obj$nowin_meanhfa),
                              "win.ind"=rep(c(1,0), each=length(timings_w)))
      p <- p + geom_line(data=sim_tmean, aes(x=timings_w, y=hfa, group=factor(win.ind)), size=1)
      
    }

    if(return_p){
      return(p)
    } else {
      print(p)
    }
  }
  
  if (want_data){
    return(list("yhat_df" = yhat_df,
                "gams_show" = gams_show,
                "params_mat" = params_mat,
                "coef_mat" = coef_mat))
  }
}












# vis_gamfits_comp(gam_mat, pmat, n_show_each=1, simdat_obj....----
vis_gamfits_comp <- function(gam_mat, pmat, n_show_each=1, simdat_obj,
                             gam_js_structure, des_mat_f, ndz, bdeg, pen_ord, 
                             lam, g, linspl_inds_obj, 
                             maintext, alph=.2,
                             plot_fits=TRUE, return_p=TRUE, timings_w=seq(-900, 1900, 50)){
  
  simdat <- simdat_obj$simdat
  if(missing(g)) g <- nrow(simdat)
  best_n_inds <- apply(pmat, 2, function(X) tail(order(X), n_show_each))
  
  all_df <- NULL
  
  for (i in 1:ncol(pmat)){
    
    type=colnames(pmat)[i]
    
    method <- ifelse(type=="Zell", "Zellner", "posterior")
    if(type=="klmin") {
      l_func<-min
    } else if(type=="klave") {
      l_func<-mean
    } else if(type=="klxtx") {
      l_func<-"xtx"
    } else {
      l_func<-max
    }
    
    gamfit <- vis_gamfits(gam_mat=gam_mat[best_n_inds[,i],], simdat=simdat, gam_js_structure=gam_js_structure,
                          des_mat_f=des_mat_f, ndz=ndz, bdeg=bdeg, pen_ord=pen_ord,
                          num_show=n_show_each, 
                          method=method,
                          lam=lam, g=g, linspl_inds_obj=linspl_inds_obj, l_func=l_func,
                          want_data=TRUE, want_plots=F)$yhat_df
    gamfit$source <- type
    all_df <- rbind(all_df, gamfit)
  }
  
  all_df$grp <- paste0(all_df$source, all_df$fit_num, all_df$win.ind)
  
  meandf <- data.frame("hfa"=c(simdat_obj$win_meanhfa, 
                               simdat_obj$nowin_meanhfa),
                       "win.ind"=rep(c(1,0), each=length(simdat_obj$win_meanhfa)),
                       "timings_w"= rep(timings_w, 2))
  if (missing(maintext)) maintext <- "comparison of fits from different variants"
  
  p <- ggplot() + 
    geom_line(data=simdat, 
              aes(y=hfa, x=timings_w, group=trial), 
              alpha=.03) +
    geom_line(data=all_df, aes(x=z, y=yhat, group=grp, color=source), size=2, alpha=alph) + 
    geom_line(data=meandf, 
              aes(y=hfa, x=timings_w, group=win.ind),
              size=1, linetype="dashed") + 
    labs(title=maintext,
         subtitle=paste0("best ", n_show_each, " models by posterior prob"))
  
  if (plot_fits) print(p)
  
  if(return_p){
    return(list("simdat"=simdat,
                "p"=p,
                "all_df"=all_df,
                "meandf"=meandf))
  } else {
    return(list("simdat"=simdat,
                "all_df"=all_df,
                "meandf"=meandf))
  }
}

####                   #### ----
####    MOMBF funcs    #### ----
####                   #### ----
# function to change beta-binomial prior probabilities to transition prior probabilities
# usage: multiply betabinom prior probs by this function's output to obtain transition prior probs
bbinom_to_ptrans <- function(gam, gam_js_structure, linspl_inds_obj, p_trans=.1, want_log=T){
  gam_js <- gam_js_from_gam_structure(gam=gam, gam_js_structure=gam_js_structure)
  bbinom_prior <- p_gam_betabinom(gam_js=gam_js, want_log=T, linspl_inds_obj=linspl_inds_obj)
  trans_prior <- p_gam_trans(gam_js=gam_js, want_log=T, linspl_inds_obj=linspl_inds_obj, p_trans=p_trans)
  if(want_log){
    return(trans_prior-bbinom_prior)
  } else {
    return(exp(trans_prior-bbinom_prior))
  }
}

normalize_logposts <- function(logposts){
  #takes log-posterior probs
  #returns probabilities after norming to sum to 1
  lp_scaled <- logposts-max(logposts)
  lp_normed <- exp(lp_scaled)/sum(exp(lp_scaled))
  return(lp_normed)
}



extract_mombf <- function(momfit, n_mods=NA){
  postmomfit <- postProb(momfit) #posterior model probabilities
  
  #extract posterior model probabilities and models
  posts <- postmomfit$pp
  mods <- postmomfit$modelid
  if(!is.na(n_mods)){
    mods <- mods[1:n_mods]
    posts <- posts[1:n_mods]
  }
  
  mods_list <- lapply(mods, function(X) as.numeric(strsplit(x=as.character(X), split=",")[[1]]))
  mod_mat <- matrix(0, nrow=length(mods_list), ncol=length(momfit$margpp))
  for(i in 1:length(mods_list)){
    mod_mat[i, mods_list[[i]]] <- 1
  }
  
  # calculate PIPs
  pips <- colSums(mod_mat*posts)
  return(list("posts"=posts,
              "mod_mat"=mod_mat,
              "pips"=pips))
}


bb_to_tr_mombf <- function(momfit_extd, n_mods=NA, gam_js_structure, linspl_inds_obj, p_trans=.1, modelSel_output){
  # provide either the modelSelection() output as momfit
  # or (faster) the extracted output from extract_mombf(momfit)
  
  if (missing(momfit_extd)){
    momfit_extd <- extract_mombf(modelSel_output)
  }
  
  #extract posterior model probabilities and models
  posts_bb <- momfit_extd$posts
  mod_mat <- momfit_extd$mod_mat
  
  bb_to_tr_log <- apply(mod_mat, 1, function(X)
    bbinom_to_ptrans(gam=X, gam_js_structure, linspl_inds_obj, p_trans=p_trans, want_log=TRUE))
  
  posts_tr_log <- log(posts_bb) + bb_to_tr_log
  # posts_tr <- normalize_logposts(posts_tr_log)
  
  # bb_to_tr <- apply(mod_mat, 1, function(X) 
  #   bbinom_to_ptrans(gam=X, gam_js_structure, linspl_inds_obj, p_trans=p_trans, want_log=FALSE))
  # posts_tr <- posts_bb * bb_to_tr_log
  # posts_tr <- posts_tr/sum(posts_tr)
  
  posts_tr_log <- posts_tr_log-max(posts_tr_log) # rescale to avoid underflow
  posts_tr <- exp(posts_tr_log)/sum(exp(posts_tr_log))
  
  #subset to best n_mods models
  if(is.na(n_mods)){
    best_tr_inds <- rev(order(posts_tr))
  } else {
    best_tr_inds <- rev(order(posts_tr))[1:n_mods]
  }
  mod_mat <- mod_mat[best_tr_inds, ]
  posts_tr <- posts_tr[best_tr_inds]
  
  # calculate PIPs
  pips_tr <- colSums(mod_mat*posts_tr)
  
  return(list("posts"=posts_tr,
              "mod_mat"=mod_mat,
              "pips"=pips_tr))
}


quantile_row <- function(postprobs, topXpercent = .9, top_n_mods = 10){
  min(
    length(postprobs$pp),
    max(
      top_n_mods,
      sum(cumsum(postprobs$pp) < topXpercent) + 1
    )
  )
}


# top X percent post prob models, FDRs for MAP and Median model ----
extract_mombf_metrics <- function(ms_obj, includevars, nullvars, topXpercent = .9){

  ## top X percent post prob models, FDRs for MAP and Median model
  
  mod_pp <- postProb(ms_obj)
  margpp <- ms_obj$margpp
  ignore <- which(includevars) # always in model
  
  endrow <- quantile_row(
    postprobs = mod_pp,
    topXpercent = topXpercent
  )
  top_pp_mods <- mod_pp[1:endrow, ]  
  
  # generate gammas for top_pp_mods
  top_pp_probs <- top_pp_mods[,3]
  top_pp_modlist <- sapply(
    top_pp_mods[,1], 
    function(X) unlist(as.numeric(strsplit(x = as.character(X), split = ",")[[1]]))
  )

  
  
  top_pp_gam_mat <- do.call(
    rbind,
    lapply(
      top_pp_modlist, 
      function(X){
        vec <- rep(0, length(includevars))
        vec[X] <- 1
        length(vec)
        return(vec)
      }
    )
  )
  rownames(top_pp_gam_mat) <- NULL
  
  # metrics
  MAP_mod <- as.numeric(
    unlist(
      strsplit(x = as.character(top_pp_mods[1, 1]), split = ",")[[1]]
    )
  )
  
  MED_mod <- which(margpp >= 0.5)
  
  MAP <- setdiff(MAP_mod, ignore)
  MED <- setdiff(MED_mod, ignore)
  
  BFDR_MAP <- ifelse(
    length(MAP) == 0, 
    0,
    mean(1 - margpp[MAP])
  )
  FDR_MAP <- ifelse(
    length(MAP) == 0, 
    0, 
    sum(length(intersect(MAP, nullvars))) / length(MAP)
  )
  
  BFDR_MED <- ifelse(
    length(MED) == 0,
    0,
    mean(1 - margpp[MED])
  )
  FDR_MED <- ifelse(
    length(MED) == 0, 
    0,
    sum(length(intersect(MED, nullvars))) / length(MED)
  )
  
  return(
    list(
      "top_pp_mods" = top_pp_mods,
      "top_pp_probs" = top_pp_probs,
      "top_pp_gam_mat" = top_pp_gam_mat,
      "margpp" = margpp,
      "MAP_mod" = MAP_mod,
      "MED_mod" = MED_mod,
      "BFDR_MAP" = BFDR_MAP,
      "FDR_MAP" = FDR_MAP,
      "BFDR_MED" = BFDR_MED,
      "FDR_MED" = FDR_MED
    )
  )
}


# extract_tr_mombf_metrics(tr_mombf_obj, includevars, nullvars .... ----
extract_tr_mombf_metrics <- function(tr_mombf_obj, includevars, nullvars, topXpercent = 0.9, top_n_mods = 10){

  ## top X percent post prob models, FDRs for MAP and Median model
  margpp <- tr_mombf_obj$pips
  ignore <- which(includevars) # always in model
  
  endrow <- min(
    length(tr_mombf_obj$posts),
    max(
      top_n_mods,
      sum(which(cumsum(tr_mombf_obj$posts) < topXpercent)) + 1
    )
  )

  
  top_pp_probs <- tr_mombf_obj$posts[1:endrow]  
  top_pp_gam_mat <- tr_mombf_obj$mod_mat[1:endrow,]
  
  # metrics
  MAP_mod <- which(top_pp_gam_mat[1, ]==1)
  
  MED_mod <- which(margpp >= 0.5)
  
  MAP <- setdiff(MAP_mod, ignore)
  MED <- setdiff(MED_mod, ignore)
  
  BFDR_MAP <- ifelse(
    length(MAP) == 0, 
    0,
    mean(1 - margpp[MAP])
  )
  FDR_MAP <- ifelse(
    length(MAP) == 0, 
    0, 
    sum(length(intersect(MAP, nullvars))) / length(MAP)
  )
  
  BFDR_MED <- ifelse(
    length(MED) == 0,
    0,
    mean(1 - margpp[MED])
  )
  FDR_MED <- ifelse(
    length(MED) == 0, 
    0,
    sum(length(intersect(MED, nullvars))) / length(MED)
  )
  
  return(
    list(
      "top_pp_probs" = top_pp_probs,
      "top_pp_gam_mat" = top_pp_gam_mat,
      "margpp" = margpp,
      "MAP_mod" = MAP_mod,
      "MED_mod" = MED_mod,
      "BFDR_MAP" = BFDR_MAP,
      "FDR_MAP" = FDR_MAP,
      "BFDR_MED" = BFDR_MED,
      "FDR_MED" = FDR_MED
    )
  )
}


####                   #### ----
####    MA(1) funcs    #### ----
####                   #### ----


balance_x1 <- function(x1, group1_code = 0, group2_code = 1, new_group2_code = 1){
  # check all items in x1 are either group1_code or group2_code
  if (!all(x1 %in% c(group1_code, group2_code))){
    warning("all items in x1 must be coded with either group1_code or group2_code")
    stop()
  }
  n_group1 <- sum(x1 == group1_code)
  n_group2 <- sum(x1 == group2_code)
  new_group1_code <- - new_group2_code * n_group2 / n_group1
  
  new_x1 <- ifelse(
    x1 == group1_code,
    new_group1_code,
    new_group2_code
  )
  
  return(new_x1)
}


# get_arima_phi_est(y, X, n_trials, .... ) ----
get_arima_phi_est <- function(
  y, X, n_trials, 
  want_acfs = FALSE, 
  want_auto.arima = F, 
  order_vec = c(0,0,1),
  get_err_cov = FALSE){
  require(forecast)
  olsfit <- lm(y ~ -1 + X)
  if (want_acfs){
    acf(olsfit$residuals)
    pacf(olsfit$residuals)
    acf(diff(olsfit$residuals))
  }
  
  ## concatenate trials with NAs in between (per Rob Hyndman's suggestion)
  resmat <- matrix(olsfit$residuals, nrow = n_trials, byrow = FALSE)
  namat <- matrix(NA, nrow = n_trials, ncol = ncol(resmat))
  arima_mat <- rbind(resmat, namat)
  arima_vec <- as.vector(arima_mat)
  
  res <- list()
  if (want_auto.arima){
    auto_fit <- auto.arima(ts(arima_vec))
    res$auto_coef <- auto_fit$coef
  }
  
  fitma <- arima(ts(arima_vec), order = order_vec, include.mean = F)
  res$spec <- fitma$coef
  
  if (get_err_cov){
    res$cov <- fitma$model$Pn
  }
  
  return(res)
}

# phi_gridsearch(des_mat_f, y, gam_mod=NULL, bcols_all, ...) ----
# phi_gridsearch uses the full model to find the optimal MA1 parameter
phi_gridsearch <- function(des_mat_f, y, gam_mod=NULL, bcols_all, 
                           n_obs_allsubj = 57, 
                           n_subj = 179,
                           timings_w = seq(-900, 1900, 50),
                           phi_grid = seq(-1, 1, length.out = 101),
                           ndz0, b0deg,
                           ndz1, b1deg){
  st_time <- Sys.time()
  gam_full <- rep(1, ndz0 + b0deg + ndz1 + b1deg)
  gam_js_structure <- list(
    "B0" = rep(1, ndz0 + b0deg),
    "B1" = rep(1, ndz1 + b1deg)
  )
  
  linspl_inds_obj <- linspl_inds_func(gam_js_structure)
  
  P_full <- make_P_full(gam_js=gam_js_structure, bdeg=c(b0deg, b1deg), pen_order=c(max(2, b0deg), max(2,b1deg))-1)
  
  # under gam_mod
  if(is.null(gam_mod)){
    gam_mod <- gam_full
  }
  des <- des_mat_f[, gam_mod==1]
  P <- make_P_red(gam=gam_mod, P_full = P_full, intercept_needed = F)
  
  # diagnostics
  dxs <- matrix(NA, nrow = length(phi_grid), ncol = 4)
  colnames(dxs) <- c("yty", "sum(diag(BtB))", "k_prec", "k_cov")
  
  #log-MLs
  lml_by_phi <- matrix(NA, nrow = length(phi_grid), ncol = 6)
  rownames(lml_by_phi) <- phi_grid
  
  # microbench results ~ 5 seconds
  for (p in 1:length(phi_grid)){
    y_ssi <- left_mult_sqrt_Sigma_inv(y, phi_grid[p], n_obs_allsubj = n_obs_allsubj, n_subj = n_subj)
    B_ssi <- left_mult_sqrt_Sigma_inv(des, phi_grid[p], n_obs_allsubj = n_obs_allsubj, n_subj = n_subj)
    
    y_ssi <- scale(y_ssi, center = FALSE)
    B_ssi <- scale(B_ssi, center = FALSE)
    
    yty <- sum(y_ssi^2)
    BtB <- t(B_ssi)%*%B_ssi
    
    # Zellner
    Lam_Z <- 1/(nrow(des)) * BtB
    
    # group-Zellner
    Lvec_f <- unlist(
      lapply(
        linspl_inds_obj$gam_js_enum,
        function(X) rep(sum(gam_mod[X]), length(X))
      )
    )
    Lvec <- Lvec_f[gam_mod==1]
    L.5 <- diag(sqrt(Lvec))
    Lam_gZ <- L.5%*%Lam_Z%*%L.5
    
    # block-scaled Zellner-P mixture
    k_prec <- sum(diag(BtB)) / sum(diag(P))
    k_cov  <- sum(1/eigen(P+1E-3*diag(nrow(P)))$values) / sum(1/eigen(BtB)$values)
    
    # #- 50-50 ZP mixture, entire precision block-scaled by dimension, k_prec and k_cov
    # #  - Lambda_0 = 1/g  L  ( (1-lam)XtX + lam*k*P)
    # #    - L = diagonal matrix with coef group dimension on diagonal
    Lam_block_kprec <- 1/nrow(B_ssi) * L.5%*%(  1/2*(BtB) + 1/2*k_prec * P  )%*%L.5
    Lam_block_kcov  <- 1/nrow(B_ssi) * L.5%*%(  1/2*(BtB) + 1/2*k_cov * P  )%*%L.5
    
    # ridge
    ridge_kprec <- 1/nrow(B_ssi) * L.5%*%(  1/2 * diag(nrow(BtB)) + 1/2*k_prec * P  )%*%L.5
    ridge_kcov <- 1/nrow(B_ssi) * L.5%*%(  1/2 * diag(nrow(BtB)) + 1/2*k_cov * P  )%*%L.5
    
    Lamlist <- list(
      "Zell" = Lam_Z,
      "grp_Zell" = Lam_gZ,
      "block_prec" = Lam_block_kprec,
      "block_cov" = Lam_block_kcov,
      "ridge_kprec" = ridge_kprec,
      "ridge_kcov" = ridge_kcov 
    )
    # lmls <- rep(NA, length(Lamlist))
    
    lmls <- unlist(lapply(Lamlist, function(Lam)
      lml_lastcalc(y=y_ssi, B=B_ssi, Lam=Lam, yty=yty, BtB=BtB, n=length(y_ssi))))
    
    dxs[p,] <- c(yty, sum(diag(BtB), k_prec, k_cov))
    lml_by_phi[p,] <- lmls
  }
  colnames(lml_by_phi) = names(Lamlist)
  
  optim_phi <- phi_grid[apply(lml_by_phi, 2, function(X) which(X ==max(X)))]
  names(optim_phi) <- names(Lamlist)
  print("optimal phi:")
  print(optim_phi)
  
  ggp <- ggplot(melt(lml_by_phi)) + 
    geom_line(aes(y = value, x = Var1, color = Var2)) + 
    labs(
      title = "log-ml ~ phi", 
      y = "lml"
    )
  
  print(paste0("phi gridsearch finished in ", round(Sys.time() - st_time), 2))
  
  return(list(
    "plot" = ggp,
    "diagnostics" = dxs,
    "lml_by_phi" = lml_by_phi,
    "optim_phi" = optim_phi
  ))
}




# make_sqrt_mat_inv(mat)----
sqrt_mat_inv <- function(mat){
  # make $\Sig^{-1/2}$
  # this takes a long time for large matrices and may fail for very large matrices
  eig <- eigen(mat)
  V <- eig$vectors
  sqrt_mat_inv <- V %*% diag(1 / sqrt(eig$values)) %*% t(V)
  return(sqrt_mat_inv)
}

# make_MA1Sigma_i(phi, n_obs_i)----
# make_MA1Sigma_i constructs the MA1 covariance ONLY for observation group i given phi and # observations in group i
make_MA1Sigma_i <- function(phi, n_obs_i){
  # for obs on a regular grid
  # make MA1 covariance for a single trial / subject
  #     phi: MA1 coef
  #     n_obs_i:  # obs for subject i
  require(Matrix)
  diags <- list(
    rep(1 + phi^2, n_obs_i),
    rep(phi, n_obs_i)
  )
  Sig_i <- cov2cor(
    Matrix::bandSparse(
      n_obs_i, 
      k = 0:1, 
      diag = diags, 
      symm=TRUE
    )
  )
  
  return(Sig_i)
}


# BROKEN ----- make_MAkSigma_i(phi, n_obs_i) ----
make_MAkSigma_i <- function(phi, n_obs_i){
  diags <- list(
    rep(
      1 + sum(phi^2),
      n_obs_i
    ),
    rep(phi[1], n_obs_i)
  )
  
  
  if (length(phi)>1){
    for (k in 1:(length(phi)-1)){
      theta_tk <- lag(phi, k)
      theta_tk[1:(k-1)] <- 0
      theta_tk[k] <- 1
      diags[[k+1]] <- rep(
        sum(phi*theta_tk),
        n_obs_i
      )
    }
  }

  Sig_i <- cov2cor(
    Matrix::bandSparse(
      n_obs_i, 
      k = 0:(length(diags)-1), 
      diag = diags, 
      symm=TRUE
    )
  )

  return(Sig_i)
}

# make_MAqSigma_i(phi, n_obs_i)
make_MAqSigma_i <- function(phi, n_obs_i){
    phi = c(1, phi)
    diags <- list(
      rep(sum(phi^2), n_obs_i)
    )
      
      
    for (tau in 1:(length(phi)-1)){
      acf_tau = 0
      for (i in 1:(length(phi)-tau)){
        acf_tau = acf_tau + phi[i] * phi[i+tau]
      }
      diags[[tau+1]] <- rep(acf_tau, n_obs_i)
    }
    
    Sig_i <- cov2cor(
      Matrix::bandSparse(
        n_obs_i, 
        k = 0:(length(diags)-1), 
        diag = diags, 
        symm=TRUE
      )
    )
    return(Sig_i)
  }



# left_mult_sqrt_Sigma_inv(mat, phi, n_obs_allsubj, n_subj) ----
# left_mult_sqrt_Sigma_inv only works when observation groups are all the same size and on a grid. 
# calculations are done in blocks (assumes independence between observation groups)
left_mult_sqrt_Sigma_inv <- function(mat, phi, n_obs_allsubj, n_subj, diagnostics = FALSE){
  # multiplication is done in blocks (by trial)
  # since Sigma_inv would be block diagonal
  Sigma_j <- make_MA1Sigma_i(phi, n_obs_i = n_obs_allsubj)
  sqrt_S_inv <- sqrt_mat_inv(Sigma_j)
  
  # check dims
  if (is.vector(mat)){
    if (length(mat) != n_obs_allsubj*n_subj){
      cat("n_obs_allsubj should be number of obs per trial")
      stop()      
    }
  } else if (nrow(mat) != n_obs_allsubj*n_subj){
    cat("n_obs_allsubj should be number of obs per trial")
    stop()
  }
  
  
  for (j in 1:n_subj){
    start_ind <- ((j - 1) * n_obs_allsubj + 1) 
    end_ind <- j * n_obs_allsubj
    
    if (is.vector(mat)){
      mat[start_ind:end_ind] <- sqrt_S_inv %*% mat[start_ind:end_ind]
    } else {
      mat[start_ind:end_ind, ] <- sqrt_S_inv %*% mat[start_ind:end_ind, ]
    }
  }
  
  if (diagnostics){
    return(
      list(
        "Sigma_j" = Sigma_j,
        "sqrt_Sigma_j_inv" = sqrt_S_inv,
        "mat" = mat
      )
    )
  } else {
    return(mat)
  }
}















