# R 3.5.1 で実行確認
# import libraries
library(nnet)
library(MASS)
library(mxnet)

train <- read.csv("data/train.csv", header = TRUE)
test <- read.csv("data/test.csv", header = TRUE)
train <- data.matrix(train) 
test <- data.matrix(test)
train.x <- train[,-1]
train.y <- train[,1]
test_org <- test
# test <- test[,-1]
train.x <- t(train.x/255) # [0, 255] -> [0, 1]
test <- t(test/255)

# preparing training
mx.set.seed(0)
devices <- mx.cpu()
tic <- proc.time()


training_mnist_cnn = function(dropout, activate_fn) {
  # input layer
  data <- mx.symbol.Variable("data")
  
  # hidden layer 1 
  conv1 <- mx.symbol.Convolution(data=data, kernel=c(5, 5), num_filter=20)
  tanh1 <- mx.symbol.Activation(data=conv1, act_type=activate_fn)
  pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2, 2),
                             stride=c(2, 2))
  
  if (dropout) {
    drop1 <- mx.symbol.Dropout(data=pool1, p=0.5)
  } else {
    drop1 <- pool1
  }
  
  # hidden layer 2
  conv2 <- mx.symbol.Convolution(data=drop1, kernel=c(5,5), num_filter=50)
  tanh2 <- mx.symbol.Activation(data=conv2, act_type=activate_fn)
  pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2, 2),
                             stride=c(2, 2))
  
  if (dropout) {
    drop2 <- mx.symbol.Dropout(data=pool2, p=0.5)
  } else {
    drop2 <- pool2
  }
  
  # fully connected layer 1
  flatten <- mx.symbol.Flatten(data=drop2)
  fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
  tanh3 <- mx.symbol.Activation(data=fc1, act_type=activate_fn)
  
  if (dropout) {
    drop3 <- mx.symbol.Dropout(data=tanh3, p=0.5)
  } else {
    drop3 <- tanh3
  }
  
  # fully connected layer 2
  fc2 <- mx.symbol.FullyConnected(data=drop3, num_hidden=10)
  
  # output layer
  lenet <- mx.symbol.SoftmaxOutput(data=fc2)
  
  # preparing train/test data
  train.array <- train.x
  dim(train.array) <- c(28, 28, 1, ncol(train.x))
  
  test.array <- test 
  dim(test.array) <- c(28, 28, 1, ncol(test))
  
  # preparing training
  mx.set.seed(0)
  devices <- mx.cpu()
  tic <- proc.time()
  
  # training model
  model.CNNtanhDrop <- mx.model.FeedForward.create(lenet, X=train.array,
                                                   y=train.y, ctx=devices, num.round = 30, array.batch.size = 100,
                                                   learning.rate=0.05, momentum=0.9, wd=0.00001,
                                                   eval.metric=mx.metric.accuracy,
                                                   batch.end.callback = mx.callback.log.train.metric(100))
  print(proc.time() - tic)
  model.CNNtanhDrop
}

model.CNNtanhDrop = training_mnist_cnn(TRUE, "relu")

preds <- predict(model.CNNtanhDrop, test.array, ctx=devices)
pred.label <- max.col(t(preds)) -1

submission <- data.frame(ImageId=1:ncol(test),
                         Label=pred.label)

write.csv(submission, file='submission.csv', row.names=FALSE, quote=FALSE)

# 結果
# 1411 位 elect 12/19/2018
# 0.99157