# R 3.5.1 で実行確認
# import libraries
library(nnet)
library(MASS)

# レポート課題1.1
# 1)
# 隠れ素子の数を一つづつ増やし、10回の学習で10回とも正しく識別出来るようになった隠れ素子の数を求めなさい。
# 2)
# 隠れ素子の数によって誤識別率の平均がどのように変化するのかをグラフで示しなさい。
# 3)
# 隠れ素子が1個の場合に得られた学習結果について、結合係数の大きさの分布を示しなさい。
# 4) 
# 10回とも正しく識別できた場合の学習結果について、結合係数の大きさの分布を示し、隠れ素子が1個の場合と比較検討しなさい。
hidden = c(1:40)
iter = c(1:10)
trave = rep(0, length(hidden))
res_s = c()
tr = matrix(0, length(iter), length(hidden))
z = 0
for (i in 1:length(hidden)) {
  for (j in 1:length(iter)) {
	res_s = c(list(nnet(classes~., data=xor, size=hidden[i], rang=0.1)), res_s)
	out = predict(res_s[i][[1]], xor, type="class")
	tr[j, i] = mean(out != xor$classes)
  }
  trave[i] = mean(tr[,i])
  if(trave[i] == 0.0 && z == 0) {
  	z = i
  }
}
res_s = rev(res_s)
# 1) 37
z
# 2) 1-1-2.png
plot(hidden, trave, type="b", lty=1, lwd=2)
# 3) 1-1-3.png
hist(res_s[1][[1]]$wts, breaks=seq(-0.1, 0.1, 0.04), freq=TRUE)
# 4) 1-1-4.png
hist(res_s[z][[1]]$wts, breaks=seq(-0.1, 0.1, 0.04), freq=TRUE)

# レポート課題１．２
# データの用意
ir <- data.frame(rbind(iris3[,,1], iris3[,,2], iris3[,,3]),
			species=factor(c(rep("sv", 50), rep("c", 50),
			rep("sv", 50))))
samp <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))
# 1)
# decay=0 に設定し、隠れ素子の数を１にして学習データを用いて１０回学習し、学習データに対する誤識別率の平均と、テストデータに対する誤識別率の平均を求めなさい。
# 隠れ素子の数を１０まで１つずつ増やして同じことを行い、隠れ素子の数に対する再代入誤りと汎化誤差の変化をグラフ化しなさい。
# 2)
# 再代入誤りが一番小さな場合の結合係数の分布と、汎化誤差が一番小さな場合の結合係数の分布を比較検討しなさい。
# 3)
# decay=0.01 にして同様の実験を行い、隠れ素子数に対する再代入誤りと汎化誤差の変化をグラフ化しなさい。
# 4)
# 再代入誤りが一番小さな場合の結合係数の分布と、汎化誤差が一番小さな場合の結合係数の分布を、decay=0 の場合と比較しなさい。
hidden = c(1:10)
iter = c(1:10)
decay = 0
trave_learn = rep(0, length(hidden))
trave_test = rep(0, length(hidden))
res_s = c()
tr_learn = matrix(0, length(iter), length(hidden))
tr_test = matrix(0, length(iter), length(hidden))
for (i in 1:length(hidden)) {
  for (j in 1:length(iter)) {
  	res_s = c(list(nnet(species~., data=ir[samp,], size=hidden[i], rang=0.5, decay=decay, maxit=200)), res_s)
	out_learn = predict(res_s[i][[1]], ir[samp,], type="class")
	out_test = predict(res_s[i][[1]], ir[-samp,], type="class")
	tr_learn[j, i] = mean(out_learn != ir[samp,]$species)
	tr_test[j, i] = mean(out_test != ir[-samp,]$species)
  }
  trave_learn[i] = mean(tr_learn[, i])
  trave_test[i] = mean(tr_test[, i])
}
res_s = rev(res_s)
# 1) 1-2-1-1.png
plot(hidden, trave_learn, type="b", lty=1, lwd=2)
# 1) 1-2-1-2.png
plot(hidden, trave_test, type="b", lty=1, lwd=2)
# 2) 
which.min(trave_learn) # 9
which.min(trave_test)  # 10
# 2) 1-2-2-1.png
hist(res_s[which.min(trave_learn)][[1]]$wts, breaks=seq(-20, 20, 5), freq=TRUE)
# 2) 1-2-2-2.png
hist(res_s[which.min(trave_test)][[1]]$wts, breaks=seq(-20, 20, 5), freq=TRUE)
# 3)
hidden = c(1:10)
iter = c(1:10)
decay = 0.01
trave_learn = rep(0, length(hidden))
trave_test = rep(0, length(hidden))
res_s = c()
tr_learn = matrix(0, length(iter), length(hidden))
tr_test = matrix(0, length(iter), length(hidden))
for (i in 1:length(hidden)) {
  for (j in 1:length(iter)) {
  	res_s = c(list(nnet(species~., data=ir[samp,], size=hidden[i], rang=0.5, decay=decay, maxit=200)), res_s)
	out_learn = predict(res_s[i][[1]], ir[samp,], type="class")
	out_test = predict(res_s[i][[1]], ir[-samp,], type="class")
	tr_learn[j, i] = mean(out_learn != ir[samp,]$species)
	tr_test[j, i] = mean(out_test != ir[-samp,]$species)
  }
  trave_learn[i] = mean(tr_learn[, i])
  trave_test[i] = mean(tr_test[, i])
}
res_s = rev(res_s)
# 3) 1-2-3-1.png
plot(hidden, trave_learn, type="b", lty=1, lwd=2)
# 3) 1-2-3-2.png
plot(hidden, trave_test, type="b", lty=1, lwd=2)
# 4)
which.min(trave_learn) # 4
which.min(trave_test) # 4
# 4) 1-2-4-1.png
hist(res_s[which.min(trave_learn)][[1]]$wts, breaks=seq(-6, 6, 1), freq=TRUE)
# 4) 1-2-4-2.png
hist(res_s[which.min(trave_test)][[1]]$wts, breaks=seq(-6, 6, 1), freq=TRUE)

