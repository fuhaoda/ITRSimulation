rm(list=ls());
source("~/Documents/CodeDev/ITRSimulation/RSim/ITRFit.R")
set.seed(1234);

n <- 1000
p <- 20
nSim <- 100
nTrt <- 2 #number for treatment
nMethod <- 2 #number of method to compare

resM <- matrix(NA, nrow=nSim, ncol=nMethod)
OptM <- matrix(NA, nrow=n,ncol=nTrt)
regretM <- matrix(NA, nrow=nSim, ncol=nMethod)

for (iter in 1:nSim) {
  print(paste("Current Simulation Step:", iter))
  X <- matrix(rnorm(n * p), nrow = n)
  A <- 1 + rbinom(n, 1, 0.9)
  pdg <- (((X[, 1] - 0.2) > 0) - 0.5)
  y <- pdg * (A - 1.5) + 0.5 * rnorm(n)
 
  for(iterTrt in 1:nTrt){
    OptM[,iterTrt] <- pdg * (iterTrt - 1.5)
  }
 
  ITR.oracle <- (pdg > 0) + 1
  dataM <- data.frame(y, A, X)
  res <- ITRFit(dataM)
  rf.err <- sum(ITR.oracle != res$ITR.rf) / n
  xgboost.err <- sum(ITR.oracle != res$ITR.xgboost) / n
  resM[iter,] <- c(rf.err,xgboost.err)
  regretM[iter,1] <- mean(t(OptM)[0:(n-1)*nMethod + ITR.oracle])- mean(t(OptM)[0:(n-1)*nMethod + res$ITR.rf])
  regretM[iter,2] <- mean(t(OptM)[0:(n-1)*nMethod + ITR.oracle])- mean(t(OptM)[0:(n-1)*nMethod + res$ITR.xgboost])
}





  
  
  
