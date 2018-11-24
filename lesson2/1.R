hidden=c(1:35)
iter=c(1:10)
trave=rep(0, length(hidden))
tr=matrix(0, length(iter), length(hidden))
for(i in 1:length(hidden)){
  for(j in 1:length(iter)){
    res=nnet(classes~., data=xor, size=hidden[i], rang=0.1)
    out=predict(res, xor, type="class")
    tr[j, i]=mean(out != xor$classes)
  }
  trave[i]=mean(tr[, i])
}

trave
# [1] 0.525 0.500 0.500 0.500 0.525 0.500 0.475 0.500 0.400 0.475 0.450 0.475
# [13] 0.450 0.350 0.350 0.250 0.350 0.200 0.325 0.250 0.150 0.300 0.200 0.200
# [25] 0.250 0.250 0.100 0.050 0.100 0.050 0.075 0.000 0.100 0.100 0.000

# x<-1:length(trave)
# plot(x, trave, type="b", lty=1, lwd=2)

# 3
hidden_size=1
for(j in 1:length(iter)){
  res=nnet(classes~., data=xor, size=hidden_size, rang=0.1)
  out=predict(res, xor, type="class")
  tr[j, i]=mean(out != xor$classes)
}
trave_=mean(tr[, i])
hist(res$wts, breaks=seq(-0.5, 0.5, 0.025), freq=TRUE)

## #### 4

hidden=c(1:35)
iter=c(1:10)
trave=rep(0, length(hidden))
tr=matrix(0, length(iter), length(hidden))
for(i in 1:length(hidden)){
  for(j in 1:length(iter)){
    res=nnet(classes~., data=xor, size=hidden[i], rang=0.1)
    out=predict(res, xor, type="class")
    tr[j, i]=mean(out != xor$classes)
  }
  if(mean(tr[, i]) == 0.0){
  	break
  } 
}
hist(res$wts, breaks=seq(-20, 80, 1), freq=TRUE)  
