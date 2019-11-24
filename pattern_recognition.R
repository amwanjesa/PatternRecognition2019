library(OpenImageR)
library(coop)
library(nnet)

mnist.dat <- read.csv("mnist.csv")

## First question
sparsity(as.matrix(mnist.dat[,2:ncol(mnist.dat)]))
means <- c()
means <- colMeans(mnist.dat[,2:ncol(mnist.dat)])

imageShow(matrix(as.numeric(means),nrow=28,ncol=28,byrow=T))
label_counts <- summary(as.factor(mnist.dat[,1]))
majority_label <- label_counts[order(label_counts, decreasing = TRUE)]
error_majority_label <- (nrow(mnist.dat) - as.numeric(majority_label[1])) / nrow(mnist.dat)

## Second question
ink_cost <- c()
ink_cost <- rowSums(mnist.dat[,2:ncol(mnist.dat)])
ink_mean <- rowMeans(mnist.dat[,2:ncol(mnist.dat)])
ink_sd <- apply(mnist.dat[,2:ncol(mnist.dat)], 1, sd)

ink_scaled <- scale(ink_cost)
ink_dataset <- as.data.frame(cbind(ink_scaled, as.factor(mnist.dat$label))) 
ink_model <- multinom(V2 ~ V1, ink_dataset)

## Third question: density (ink_cost) + sparsity

  
  
  
  
  