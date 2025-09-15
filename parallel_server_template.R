# run simulation ----------------------------------------------------------

set.seed(params$seed)


for(sig in params$sigs){
  gc()
  
  fname <- make_fname(
    params, 
    sig, 
    ftype = ifelse(params$testing, "_testing.Rdata", ".Rdata")
  )  
  
  seed_vec <- floor(runif(params$n_sims)*1E6)  
  seed_list <- as.list(seed_vec)
  
  # parallelize over seed_vec
  library(parallel)
  library(pbapply)
  
  
  #initialize cluster; number cores - 4
  ncpus <- detectCores()
  cl_size <- floor(ncpus/cpu_div)
  
  cl <- makeCluster(cl_size,
                    type="PSOCK",
                    outfile=paste0(fname, "_monitor.txt"))
  
  #EXPORT variables, libraries
  # export libraries
  parallel::clusterEvalQ(cl= cl, library(fdatest))
  parallel::clusterEvalQ(cl= cl, library(ggplot2))
  parallel::clusterEvalQ(cl= cl, library(here))
  parallel::clusterEvalQ(cl= cl, library(mombf))
  parallel::clusterEvalQ(cl= cl, library(Matrix))
  parallel::clusterEvalQ(cl= cl, library(dplyr))
  
  # SOURCE functions
  parallel::clusterCall(cl, function() { source(here::here("fcns", "MCMC_func_v3.41.R")) })
  
  # export variables
  parallel::clusterExport(
    cl=cl,
    envir=environment(),
    varlist=c(
      "make_x1_true",
      "binary_error",
      "grep_get",
      "get_block_ends",
      "make_bdiag",
      "make_bdiag_by_interval",
      "reshape_for_IWT",
      "get_IWT_fit",
      "balance_x1",
      "mombf_metrics_by_ytypes",
      "phi",
      "ofc_path",
      "timings_w",
      "simdat_list",
      "simdat",
      "meandat",
      "n_trials",
      "n_obs_i",
      "b0mat_obj",
      "B",
      "b1mat_obj",
      "B_x1_raw",
      "X",
      "true_x1_gamma",
      "mf",
      "sim_func",
      "fname",
      "seed_vec",
      "sig",
      "params"
    )
  )
  
  
  # Set seed
  parallel::clusterSetRNGStream(cl, iseed = 0L)
  
  
  sims <- pblapply(
    seed_list,
    function(seed) 
      # return(
      #   tryCatch(
      sim_func(
        seed = seed, 
        mf=mf, 
        sig=sig, 
        phi=phi, 
        simdat_list=simdat_list, 
        X=X, 
        b0mat_obj=b0mat_obj, 
        b1mat_obj=b1mat_obj,
        get_IWT_default = !params$testing,
        S_chol_t_provided = params$use_saved_chol
        #       ),
        #       error = function(e) e
        # )
      ),
    cl = cl
  )
  cat(" \n \n sims done \n \n")
  
  contents <- list(
    "fname" = fname,
    "sims" = sims,
    "seed_vec" = seed_vec,
    "params" = params
  )
  
  cat(" \n \n contents made \n \n ")
  save(contents, file = here::here(fname))
  
  cat(" \n \n FILE SAVED \n \n ")
  announce <- pbapply(matrix(1:4, nrow=1), 1, function(X) cat(paste0("\n \n \n  ----", fname, " SAVED (",params$n_sims, " sims)---- \n \n \n"))
                      , cl=cl)
  
  parallel::stopCluster(cl)
}


file.remove(".RData")
q("no")