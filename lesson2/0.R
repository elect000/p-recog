# import library
library(nnet)
library(MASS)

# create datas
x1<-c(0,0)
x2<-c(0,1)
x3<-c(1,0)
x4<-c(1,1)
x<-rbind(x1, x2, x3, x4)
y<-c("-1", "1", "1", "-1")
xor<-data.frame(x, classes=y)

# check data
head(xor)
#    X1 X2 classes
# x1  0  0      -1
# x2  0  1       1
# x3  1  0       1
# x4  1  1      -1

# learning with nnet
mlp<-nnet(classes~., data=xor, size=2, rang=0.5, decay=0, maxit=100)

# refer summary
summary(mlp)
# a 2-2-1 network with 9 weights
# options were - entropy fitting 
#  b->h1 i1->h1 i2->h1 
#  -2.30 -13.00  11.42 
#  b->h2 i1->h2 i2->h2 
#  23.02 -33.96  37.16 
#   b->o  h1->o  h2->o 
#  16.15  27.36 -33.99 

hist(mlp$wts, breaks=seq(-100, 100,5), freq=TRUE)

table(xor$classes, predict(mlp, xor, type="class"))
#      -1 1
#   -1  2 0
#   1   0 2
