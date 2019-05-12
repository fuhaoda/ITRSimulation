library(xgboost)
library(randomForest)


ITRFit <- function(dataM){
  
  n <- length(dataM$y) #sample size
  p <- dim(dataM)[2]-2 #minus y and A = number of parameters
  
  trt <- sort(unique(dataM$A))
  
  rf.fit <- list();
  po.rf <- matrix(NA, nrow=n,ncol = length(trt))
  ITR.rf <- rep(NA, n)
  
  
  xgboost.fit <- list();
  po.xgboost <- matrix(NA, nrow=n,ncol = length(trt))
  ITR.xgboost <- rep(NA, n)
  
  
  
  for(iter in trt){
    rf.fit[[iter]]<- randomForest(y~.-A, data=dataM[dataM$A==iter,])
    po.rf[,iter] <- predict(rf.fit[[iter]], dataM[-c(1,2)])
    xgboost.fit[[iter]]<- xgboost(data = as.matrix(dataM[dataM$A==iter,-c(1,2)]), label = dataM$y[dataM$A==iter], max_depth = 2, eta = 0.3, nthread = 2, nrounds = 1, objective = "reg:linear", verbose = 0)
    po.xgboost[,iter] <- predict(xgboost.fit[[iter]], as.matrix(dataM[-c(1,2)]))
  }
  
  
  ITR.rf <- apply(po.rf, 1, which.max)
  ITR.xgboost <- apply(po.xgboost, 1, which.max)
  
    
  result <- list(ITR.rf=ITR.rf, ITR.xgboost=ITR.xgboost)
  return(result)
}