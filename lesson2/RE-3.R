# R 3.5.1 で実行確認
# import libraries
library(nnet)
library(MASS)
library(mxnet)

# create dataset
train <- read.csv("data/short_prac_train.csv", header = TRUE)
test <- read.csv("data/short_prac_test.csv", header = TRUE)
train <- data.matrix(train) 
test <- data.matrix(test)
train.x <- train[,-1]
train.y <- train[,1]
test_org <- test
test <- test[,-1]
train.x <- t(train.x/255) # [0, 255] -> [0, 1]
test <- t(test/255)
table(train.y)

# input layer
data <- mx.symbol.Variable("data")

# hidden layer 1 
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5, 5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2, 2),
                           stride=c(2, 2))
drop1 <- mx.symbol.Dropout(data=pool1, p=0.5)

# hidden layer 2
conv2 <- mx.symbol.Convolution(data=drop1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2, 2),
                           stride=c(2, 2))
drop2 <- mx.symbol.Dropout(data=pool2, p=0.5)

# fully connected layer 1
flatten <- mx.symbol.Flatten(data=drop2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
drop3 <- mx.symbol.Dropout(data=tanh3, p=0.5)

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
                                                 learning.rate=0.05, momentum=0.9, wd=0.000001,
                                                 eval.metric=mx.metric.accuracy,
                                                 batch.end.callback = mx.callback.log.train.metric(100))
print(proc.time() - tic)
preds <- predict(model.CNNtanhDrop, test.array, ctx=devices)
pred.label <- max.col(t(preds)) -1
sum(diag(table(test_org[,1], pred.label))) / 1000
# 1)
# 1-1)
# M1 : 24
# N1 : 5000
# M2 : 12
# N2 : 5000
# 1-2)
# M3 : 8
# N3 : 5000
# M4 : 4
# N4 : 5000
# 1-3)
# 3次元配列2次元配列に変換している
# 2)
# ---------------------------------------------
# [1] Train-accuracy=0.0943999997526407
# [2] Train-accuracy=0.089199999794364
# [3] Train-accuracy=0.095800000205636
# [4] Train-accuracy=0.353800000697374
# [5] Train-accuracy=0.815199997425079
# [6] Train-accuracy=0.879399998188019
# [7] Train-accuracy=0.910400002002716
# [8] Train-accuracy=0.919399999380112
# [9] Train-accuracy=0.933800001144409
# [10] Train-accuracy=0.933000004291534
# [11] Train-accuracy=0.939599999189377
# [12] Train-accuracy=0.945799996852875
# [13] Train-accuracy=0.944800004959106
# [14] Train-accuracy=0.945000002384186
# [15] Train-accuracy=0.946200004816055
# [16] Train-accuracy=0.960200003385544
# [17] Train-accuracy=0.956600003242493
# [18] Train-accuracy=0.954999998807907
# [19] Train-accuracy=0.958400005102158
# [20] Train-accuracy=0.961600004434586
# [21] Train-accuracy=0.962000002861023
# [22] Train-accuracy=0.960800007581711
# [23] Train-accuracy=0.964000006914139
# [24] Train-accuracy=0.965400005578995
# [25] Train-accuracy=0.966400007009506
# [26] Train-accuracy=0.968800005912781
# [27] Train-accuracy=0.964800003767014
# [28] Train-accuracy=0.967800005674362
# [29] Train-accuracy=0.965600006580353
# [30] Train-accuracy=0.969400007724762
# [1] 0.986
# ---------------------------------------------

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
                                      learning.rate=0.05, momentum=0.9, wd=0.000001,
                                      eval.metric=mx.metric.accuracy,
                                      batch.end.callback = mx.callback.log.train.metric(100))
  print(proc.time() - tic)
  preds <- predict(model.CNNtanhDrop, test.array, ctx=devices)
  pred.label <- max.col(t(preds)) -1
  sum(diag(table(test_org[,1], pred.label))) / 1000
}

# 3)
training_mnist_cnn(FALSE, "tanh")
# --------------------------------------------------------------------
# [1] Train-accuracy=0.0951999997347593
# [2] Train-accuracy=0.0893999997526407
# [3] Train-accuracy=0.0931999997794628
# [4] Train-accuracy=0.353799997121096
# [5] Train-accuracy=0.841599998474121
# [6] Train-accuracy=0.91860000371933
# [7] Train-accuracy=0.951199996471405
# [8] Train-accuracy=0.96200000166893
# [9] Train-accuracy=0.970600011348724
# [10] Train-accuracy=0.980000009536743
# [11] Train-accuracy=0.984800010919571
# [12] Train-accuracy=0.991400008201599
# [13] Train-accuracy=0.992800006866455
# [14] Train-accuracy=0.994400005340576
# [15] Train-accuracy=0.996200003623962
# [16] Train-accuracy=0.995200004577637
# [17] Train-accuracy=0.996800003051758
# [18] Train-accuracy=0.998200001716614
# [19] Train-accuracy=0.999400000572205
# [20] Train-accuracy=1
# [21] Train-accuracy=1
# [22] Train-accuracy=1
# [23] Train-accuracy=1
# [24] Train-accuracy=1
# [25] Train-accuracy=1
# [26] Train-accuracy=1
# [27] Train-accuracy=1
# [28] Train-accuracy=1
# [29] Train-accuracy=1
# [30] Train-accuracy=1
# user  system elapsed 
# 549.00  251.10  163.95 
# [1] 0.982
# --------------------------------------------------------------------
# comment: over fitting
# 4)
training_mnist_cnn(FALSE, "relu")
# --------------------------------------------------------------------
# [1] Train-accuracy=0.0957999996095896
# [2] Train-accuracy=0.0893999997526407
# [3] Train-accuracy=0.0903999998420477
# [4] Train-accuracy=0.112800000011921
# [5] Train-accuracy=0.434800001382828
# [6] Train-accuracy=0.876400001049042
# [7] Train-accuracy=0.945999997854233
# [8] Train-accuracy=0.963000003099442
# [9] Train-accuracy=0.964600006341934
# [10] Train-accuracy=0.971200007200241
# [11] Train-accuracy=0.978200010061264
# [12] Train-accuracy=0.987200009822846
# [13] Train-accuracy=0.989800009727478
# [14] Train-accuracy=0.991600008010864
# [15] Train-accuracy=0.992000007629395
# [16] Train-accuracy=0.996000003814697
# [17] Train-accuracy=0.997800002098084
# [18] Train-accuracy=0.999400000572205
# [19] Train-accuracy=0.999400000572205
# [20] Train-accuracy=0.998400001525879
# [21] Train-accuracy=0.999800000190735
# [22] Train-accuracy=0.999000000953674
# [23] Train-accuracy=0.999400000572205
# [24] Train-accuracy=0.999800000190735
# [25] Train-accuracy=0.999800000190735
# [26] Train-accuracy=0.999800000190735
# [27] Train-accuracy=1
# [28] Train-accuracy=1
# [29] Train-accuracy=1
# [30] Train-accuracy=1
# user  system elapsed 
# 513.99  243.98  154.81 
# [1] 0.982
# --------------------------------------------------------------------
# 5)
training_mnist_cnn(TRUE, "relu")
# [1] Train-accuracy=0.0949999997764826
# [2] Train-accuracy=0.0893999997526407
# [3] Train-accuracy=0.0899999997764826
# [4] Train-accuracy=0.100199999809265
# [5] Train-accuracy=0.359799997210503
# [6] Train-accuracy=0.774200001955032
# [7] Train-accuracy=0.879999998807907
# [8] Train-accuracy=0.90300000667572
# [9] Train-accuracy=0.920800005197525
# [10] Train-accuracy=0.927600004673004
# [11] Train-accuracy=0.94440000295639
# [12] Train-accuracy=0.945800001621246
# [13] Train-accuracy=0.945199998617172
# [14] Train-accuracy=0.949000002145767
# [15] Train-accuracy=0.953400001525879
# [16] Train-accuracy=0.960600000619888
# [17] Train-accuracy=0.955800002813339
# [18] Train-accuracy=0.963600004911423
# [19] Train-accuracy=0.960200004577637
# [20] Train-accuracy=0.964400005340576
# [21] Train-accuracy=0.966000009775162
# [22] Train-accuracy=0.967400008440018
# [23] Train-accuracy=0.964800004959106
# [24] Train-accuracy=0.968200006484985
# [25] Train-accuracy=0.969400004148483
# [26] Train-accuracy=0.970800008773804
# [27] Train-accuracy=0.971400009393692
# [28] Train-accuracy=0.966400008201599
# [29] Train-accuracy=0.968200007677078
# [30] Train-accuracy=0.972400006055832
# user  system elapsed 
# 543.69  238.85  157.29 
# [1] 0.98
# ---------------------------------------------------------