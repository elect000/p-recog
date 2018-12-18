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
                                                 y=train.y, ctx=devices, num.round = 30, 
                                                 array.batch.size = 100,
                                                 learning.rate=0.05, momentum=0.9, wd=0.00001,
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
# [3] Train-accuracy=0.0956000000983477
# [4] Train-accuracy=0.353200001269579
# [5] Train-accuracy=0.814999997615814
# [6] Train-accuracy=0.879199998378754
# [7] Train-accuracy=0.91200000166893
# [8] Train-accuracy=0.918200002908707
# [9] Train-accuracy=0.933600000143051
# [10] Train-accuracy=0.934600001573563
# [11] Train-accuracy=0.940800001621246
# [12] Train-accuracy=0.947199996709824
# [13] Train-accuracy=0.945600001811981
# [14] Train-accuracy=0.943799996376038
# [15] Train-accuracy=0.945399998426437
# [16] Train-accuracy=0.958400007486343
# [17] Train-accuracy=0.956800009012222
# [18] Train-accuracy=0.957000002861023
# [19] Train-accuracy=0.958199999332428
# [20] Train-accuracy=0.958999999761581
# [21] Train-accuracy=0.965000002384186
# [22] Train-accuracy=0.966000006198883
# [23] Train-accuracy=0.966400002241135
# [24] Train-accuracy=0.964200004339218
# [25] Train-accuracy=0.966800007820129
# [26] Train-accuracy=0.964200004339218
# [27] Train-accuracy=0.962800003290176
# [28] Train-accuracy=0.970400004386902
# [29] Train-accuracy=0.969200010299683
# [30] Train-accuracy=0.97040000796318
# [1] 0.985
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
                                      learning.rate=0.05, momentum=0.9, wd=0.00001,
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
# [4] Train-accuracy=0.351199997663498
# [5] Train-accuracy=0.840799999237061
# [6] Train-accuracy=0.919000004529953
# [7] Train-accuracy=0.951199996471405
# [8] Train-accuracy=0.962000002861023
# [9] Train-accuracy=0.970400011539459
# [10] Train-accuracy=0.979800009727478
# [11] Train-accuracy=0.984800012111664
# [12] Train-accuracy=0.991200008392334
# [13] Train-accuracy=0.992800006866455
# [14] Train-accuracy=0.994800004959106
# [15] Train-accuracy=0.996200003623962
# [16] Train-accuracy=0.995400004386902
# [17] Train-accuracy=0.997000002861023
# [18] Train-accuracy=0.997800002098084
# [19] Train-accuracy=0.999200000762939
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
# [4] Train-accuracy=0.110599999576807
# [5] Train-accuracy=0.418600002527237
# [6] Train-accuracy=0.869200000762939
# [7] Train-accuracy=0.947000004053116
# [8] Train-accuracy=0.965000007152557
# [9] Train-accuracy=0.973400008678436
# [10] Train-accuracy=0.975400009155273
# [11] Train-accuracy=0.980800009965897
# [12] Train-accuracy=0.986800010204315
# [13] Train-accuracy=0.99100000500679
# [14] Train-accuracy=0.991400008201599
# [15] Train-accuracy=0.994200005531311
# [16] Train-accuracy=0.99300000667572
# [17] Train-accuracy=0.991000006198883
# [18] Train-accuracy=0.995400004386902
# [19] Train-accuracy=0.996800003051758
# [20] Train-accuracy=0.995800004005432
# [21] Train-accuracy=0.998200001716614
# [22] Train-accuracy=0.999800000190735
# [23] Train-accuracy=1
# [24] Train-accuracy=1
# [25] Train-accuracy=1
# [26] Train-accuracy=1
# [27] Train-accuracy=1
# [28] Train-accuracy=1
# [29] Train-accuracy=1
# [30] Train-accuracy=1
# user  system elapsed 
# 513.99  243.98  154.81 
# [1] 0.976
# --------------------------------------------------------------------
# 5)
training_mnist_cnn(TRUE, "relu")
# [1] Train-accuracy=0.0949999997764826
# [2] Train-accuracy=0.0893999997526407
# [3] Train-accuracy=0.0899999997764826
# [4] Train-accuracy=0.100200000032783
# [5] Train-accuracy=0.35619999974966
# [6] Train-accuracy=0.773400000333786
# [7] Train-accuracy=0.879000000953674
# [8] Train-accuracy=0.908400002717972
# [9] Train-accuracy=0.922000004053116
# [10] Train-accuracy=0.927400002479553
# [11] Train-accuracy=0.938999998569489
# [12] Train-accuracy=0.948400005102158
# [13] Train-accuracy=0.951800004243851
# [14] Train-accuracy=0.953199998140335
# [15] Train-accuracy=0.955200002193451
# [16] Train-accuracy=0.959800000190735
# [17] Train-accuracy=0.95600000500679
# [18] Train-accuracy=0.956400002241135
# [19] Train-accuracy=0.959000000953674
# [20] Train-accuracy=0.961200001239777
# [21] Train-accuracy=0.966800006628036
# [22] Train-accuracy=0.968200008869171
# [23] Train-accuracy=0.969600011110306
# [24] Train-accuracy=0.968600010871887
# [25] Train-accuracy=0.970800009965897
# [26] Train-accuracy=0.970800012350082
# [27] Train-accuracy=0.968200005292892
# [28] Train-accuracy=0.971000007390976
# [29] Train-accuracy=0.969400010108948
# [30] Train-accuracy=0.973800009489059
# user  system elapsed 
# 543.69  238.85  157.29 
# [1] 0.986
# ---------------------------------------------------------

