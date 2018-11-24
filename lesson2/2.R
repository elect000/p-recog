ir <- data.frame(rbind(iris3[,,1],iris3[,,2],iris3[,,3]),
species=factor(c(rep("sv",50), rep("c",50),
rep("sv",50))))
samp <- c(sample(1:50,25),sample(51:100,25),sample(101:150,25))

##
mlpir <- nnet(species~., data=ir[samp,], size=2, rang=0.5, decay=0,
maxit=200)
#  weights:  13
# initial  value 56.936160 
# iter  10 value 34.715457
# iter  20 value 34.657419
# final  value 34.657360 
# converged

table(ir$species[samp], predict(mlpir, ir[samp,], type="class"))
#     sv
#  c  25
#  sv 50

table(ir$species[-samp], predict(mlpir, ir[-samp,], type="class"))

#      sv
#   c  25
#   sv 50
##

mlpir <- nnet(species~., data=ir[samp,], size=1, rang=0.5, decay=0,
maxit=200)
table(ir$species[samp], predict(mlpir, ir[samp,], type="class"))    
table(ir$species[-samp], predict(mlpir, ir[-samp,], type="class"))   
