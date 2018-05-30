runYAPS_bi <- function(hydro_data,toa,ss,mean_BI,params = NULL,max_it = 500000){
  #reformat TOA data
  toa <- t(as.matrix(toa))
  T0 <- min(toa, na.rm=TRUE) #reshape so first TOA is 0
  toa <- toa - T0	
  toa[is.na(toa)] <- -9999 #NA's are translated to -9999
  #reformat ss
  ss <- as.matrix(ss)
  # reformat hydro locations
  Hx <- hydro_data$X
  Hy <- hydro_data$Y
  ###########################
  ## read in known data
  ###########################
  datTmb <- list(
    H = matrix(c(Hx, Hy), ncol=2),
    toa = toa,	
    ss = ss,
    nh = length(Hx),
    np = ncol(toa),
    mean_BI = mean_BI
  )
  ###########################
  ## estimate initial values
  ###########################
  
  getParams <-list(
    XY = matrix(c(mean(datTmb$H[,1]) + rnorm(datTmb$np, sd=50), mean(datTmb$H[,2]) + rnorm(ncol(datTmb$toa), sd=50)), ncol=2),	#positions
    top = na.approx(apply(datTmb$toa, 2, function(k) {mean(k[k != -9999])}), rule=2)
    )
  if (length(params) != 5){
    getParams <- append(getParams,list(
      logD_xy = -0.25,				#diffusivity of transmitter movement (D_xy in ms)
      logSigma_bi = -0.25,			#sigma  burst interval (sigma_bi in ms)
      logSigma_toa = -2,			#sigma for Gaussian 
      logScale = -3,				#scale parameter for t-distribution
      log_t_part = -3				#Mixture ratio between Gaussian and t
      ))
  }else{
    getParams <- append(getParams,params)
  }
  
  ###########################
  ## compile and run YAPS
  ###########################
  dyn.load(dynlib("yaps_bi"))
  obj <- MakeADFun(datTmb,getParams,DLL="yaps_bi",random=c("XY","top"),inner.control = list(maxit = max_it), silent=TRUE)
  opt <- nlminb(unlist(getParams[3:length(getParams)]),obj$fn,obj$gr)
  
  #Obtain parameter estimates and standard deviations
  obj$fn()
  pl <- obj$env$parList()
  jointrep <- sdreport(obj, getJointPrecision=TRUE)
  param_names <- rownames(summary(jointrep))
  sds <- summary(jointrep)[,2]
  summ <- data.frame(param=param_names, sd=sds)
  plsd <- split(summ[,2], f=summ$param)
  
  #Extract data 
  sd_xy <- matrix(plsd$XY, ncol=2)
  yapsRes <- data.frame(x=pl$XY[,1], y=pl$XY[,2], top=pl$top+T0, sd_x=sd_xy[,1], sd_y=sd_xy[,2])
  return(yapsRes)
}











