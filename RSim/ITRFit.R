library(xgboost)
library(randomForest)


ITRFit <- function(dataM){
  
  n <- length(dataM$y) #sample size
  p <- dim(dataM)[2]-2 #minus y and A = number of parameters
  
  trt <- sort(unique(dataM$A))
  
  rf.fit <- list();
  po.rf <- matrix(NA, nrow=n,ncol = length(trt))
  ITR.rf <- rep(NA, n)
  rf.err <- 1
  
  xgboost.fit <- list();
  po.xgboost <- matrix(NA, nrow=n,ncol = length(trt))
  ITR.xgboost <- rep(NA, n)
  xgboost.err <- 1
  
  
  
  for(iter in trt){
    rf.fit[[iter]]<- randomForest(y~.-A, data=dataM[dataM$A==iter,])
    po.rf[,iter] <- predict(rf.fit[[iter]], dataM[-c(1,2)])
  }
  
  ITR.rf <- apply(po.rf, 1, which.max)
  rf.err <- sum(ITR.oracle!=ITR.rf)/n
  
  result <- list(ITR.rf, rf.err)
  return(result)
}