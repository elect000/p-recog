# R 3.5.1 で実行確認
# import libraries
library(nnet)
library(MASS)
library(mxnet)

# create dataset
train <- read.csv("data/short_prac_train.csv", header = TRUE)
test <- read.csv("data/short_prac_test.csv", header = TRUE)
train <- data.matrix(train) test <- data.matrix(test)
train.x <- train[,-1]
train.y <- train[,1]
test_org <- test
test <- test[,-1]
train.x <- t(train.x/255) # [0, 255] -> [0, 1]
test <- t(test/255)
table(train.y)

# check image
image(x=seq(1:28),y=seq(1:28), matrix(train.x[,4], 28, 28)[, 28:1],
      col = gray(0:255/255))

# sample 
# network settings
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

# network training
devices <- mx.cpu()
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X = train.x, y = train.y, 
                                     initializer = mx.init.uniform(0.07),
                                     ctx = devices,
                                     num.round = 10, array.batch.size = 100,
                                     learning.rate=0.05,
                                     momentum=0.9, wd=0.00001,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

preds <- predict(model, test, ctx=devices)
pred.label <- max.col(t(preds)) -1
sum(diag(table(test_org[,1], pred.label))) / 1000
table(test_org[,1], pred.label)

# レポート課題２．１
# 1) 
# ３つの異なった乱数の種を用いて、学習データとテストデータに対する認識率を求めなさい。
# 2)
# 最初の2つの隠れ層の非線形出力関数をシグモイド関数(sigmoid) にした場合、認識率はどのようになるか。ReLUの場合と同じ条件で実験し、比較しなさい。

training_mnist <- function(seed, activate_fun) {
  # network settings 
  data <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
  act1 <- mx.symbol.Activation(fc1, name="relu1", act_type=activate_fun)
  fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
  act2 <- mx.symbol.Activation(fc2, name="relu2", act_type=activate_fun)
  fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
  softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")
  
  devices <- mx.cpu()
  mx.set.seed(seed)
  
  # training network
  model <- mx.model.FeedForward.create(softmax, X = train.x, y = train.y, 
                                       initializer = mx.init.uniform(0.07),
                                       ctx = devices,
                                       num.round = 10, array.batch.size = 100,
                                       learning.rate=0.05,
                                       momentum=0.9, wd=0.00001,
                                       eval.metric = mx.metric.accuracy,
                                       epoch.end.callback = mx.callback.log.train.metric(100))
  preds <- predict(model, test, ctx=devices)
  pred.label <- max.col(t(preds)) -1
  return(mean(test_org[,1] == pred.label))
}

# 1)
seeds = list(11, 25, 2018)

training_mnist(seeds[1][[1]], "relu")

# --------------------------------------------
# [1] Train-accuracy=0.41060000102967
# [2] Train-accuracy=0.813400003910065
# [3] Train-accuracy=0.891999999284744
# [4] Train-accuracy=0.911600003242493
# [5] Train-accuracy=0.937400004863739
# [6] Train-accuracy=0.948400005102158
# [7] Train-accuracy=0.966600004434586
# [8] Train-accuracy=0.974000008106232
# [9] Train-accuracy=0.979200007915497
# [10] Train-accuracy=0.984000010490418
# [1] 0.938
# --------------------------------------------

training_mnist(seeds[2][[1]], "relu")
# --------------------------------------------
# [1] Train-accuracy=0.426600000560284
# [2] Train-accuracy=0.818000000715256
# [3] Train-accuracy=0.873200000524521
# [4] Train-accuracy=0.902800003290176
# [5] Train-accuracy=0.933199996948242
# [6] Train-accuracy=0.950400000810623
# [7] Train-accuracy=0.961600004434586
# [8] Train-accuracy=0.96960000872612
# [9] Train-accuracy=0.979400007724762
# [10] Train-accuracy=0.98080001115799
# [1] 0.941
# --------------------------------------------

training_mnist(seeds[3][[1]], "relu")
# --------------------------------------------
# [1] Train-accuracy=0.44320000231266
# [2] Train-accuracy=0.831800000667572
# [3] Train-accuracy=0.890200002193451
# [4] Train-accuracy=0.921600000858307
# [5] Train-accuracy=0.937600003480911
# [6] Train-accuracy=0.949200004339218
# [7] Train-accuracy=0.960400002002716
# [8] Train-accuracy=0.969600001573563
# [9] Train-accuracy=0.971800007820129
# [10] Train-accuracy=0.967400006055832
# [1] 0.931
# --------------------------------------------

# 2)
training_mnist(seeds[1][[1]], "sigmoid")
# --------------------------------------------
# [1] Train-accuracy=0.0967999996244907
# [2] Train-accuracy=0.117599999085069
# [3] Train-accuracy=0.153000000119209
# [4] Train-accuracy=0.275600000321865
# [5] Train-accuracy=0.43559999704361
# [6] Train-accuracy=0.577199996709824
# [7] Train-accuracy=0.684799997806549
# [8] Train-accuracy=0.750999997854233
# [9] Train-accuracy=0.796199997663498
# [10] Train-accuracy=0.821999995708466
# [1] 0.827
# --------------------------------------------

training_mnist(seeds[2][[1]], "sigmoid")
# --------------------------------------------
# [1] Train-accuracy=0.102400000393391
# [2] Train-accuracy=0.106399999856949
# [3] Train-accuracy=0.132000000178814
# [4] Train-accuracy=0.217799999862909
# [5] Train-accuracy=0.385399999022484
# [6] Train-accuracy=0.526599999666214
# [7] Train-accuracy=0.66940000295639
# [8] Train-accuracy=0.76299999833107
# [9] Train-accuracy=0.807399994134903
# [10] Train-accuracy=0.829199995994568
# [1] 0.84
# --------------------------------------------

training_mnist(seeds[3][[1]], "sigmoid")
# --------------------------------------------
# [1] Train-accuracy=0.0975999997928739
# [2] Train-accuracy=0.106199999824166
# [3] Train-accuracy=0.129200000017881
# [4] Train-accuracy=0.204800001382828
# [5] Train-accuracy=0.416599997282028
# [6] Train-accuracy=0.578999997973442
# [7] Train-accuracy=0.695199999809265
# [8] Train-accuracy=0.759400001764297
# [9] Train-accuracy=0.807399997711182
# [10] Train-accuracy=0.839199997186661
# [1] 0.837
# --------------------------------------------