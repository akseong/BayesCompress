sim_func_data2 <- function(
    n_obs = 10000,
    d_in = 104,
    xdist = rnorm,
    flist = list(fcn1, fcn2, fcn3, fcn4),
    flist_inputs = NULL, # list of columns of x to input to each function in flist
    err_sigma = 1,
    use_cuda = FALSE
){
  # flist: list of functions.  must take one input only, either vector or matrix
  # flist_inputs: list of integer scalars/vectors indicating which columns of x each function in flist uses
  #               if left NULL, flist assumed to be univariate functions   

  xmat <- matrix(
    xdist(n = n_obs*d_in),
    nrow = n_obs
  )
  xmat <- apply(xmat, 2, scale) # normalize to mean 0, sd 1
  x <- torch_tensor(xmat)

  # # testing
  # f_2var <- function(mat){
  #   mat[, 1] * mat[, 2]
  # }
  #flist <- list(fcn1, fcn2, fcn3, f_2var, fcn4)
  #flist_inputs <- list(1, 2, 3, c(4, 5), 6)
  
  if (is.null(flist_inputs)){
    flist_inputs <- as.list(1:(length(flist)))
  }
  d_true <- length(unique(do.call(c, flist_inputs)))

  Ey <- rep(0, n_obs)
  for(j in 1:length(flist)){
    cols <- flist_inputs[[j]]
    Ey <- Ey + flist[[j]](x[, cols])
  }
  
  y <- Ey$unsqueeze_(2) + torch_normal(mean = 0, std = err_sigma, size = c(n_obs, 1))
  
  if (use_cuda){
    x <- x$to(device = "cuda")
    y <- y$to(device = "cuda")
  }
  
  return(
    list(
      "y" = y,
      "x" = x,
      "Ey" = Ey,
      "n_obs" = n_obs,
      "d_in" = d_in,
      "d_true" = d_true
    )
  )
}

