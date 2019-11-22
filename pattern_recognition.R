library(OpenImageR)
library(coop)

#mnist.dat <- read.csv("mnist.csv")
sparsity(as.matrix(mnist.dat[,2:ncol(mnist.dat)]))

means <- c()
for(i in 2:ncol(mnist.dat)){
   means[i-1] <- mean(mnist.dat[,i])
}
# 
imageShow(matrix(as.numeric(means),nrow=28,ncol=28,byrow=T))
label_counts <- summary(as.factor(mnist.dat[,1]))
majority_label <- label_counts[order(label_counts, decreasing = TRUE)]
error_majority_label <- (nrow(mnist.dat) - as.numeric(majority_label[1])) / nrow(mnist.dat)
