rm(list=ls());
source("~/Documents/CodeDev/ITRSimulation/RSim/ITRFit.R")
set.seed(1234);

n <- 1000
p <- 20
nSim <- 100

resM <- matrix(NA, nrow=nSim, ncol=2)

for (iter in 1:nSim) {
  print(paste("Current Simulation Step:", iter))
  
  X <- matrix(rnorm(n * p), nrow = n)
  A <- 1 + rbinom(n, 1, 0.9)
  pdg <- (((X[, 1] - 0.2) > 0) - 0.5)
  y <- pdg * (A - 1.5) + 0.5 * rnorm(n)
  ITR.oracle <- (pdg > 0) + 1
  dataM <- data.frame(y, A, X)
  res <- ITRFit(dataM)
  rf.err <- sum(ITR.oracle != res$ITR.rf) / n
  xgboost.err <- sum(ITR.oracle != res$ITR.xgboost) / n
  resM[iter,] <- c(rf.err,xgboost.err)
}




  
  
  
