library(OpenImageR)
library(coop)
library(nnet)
library(glmnet)
library(e1071)

mnist.dat <- read.csv("mnist.csv")

## First question
sparse <- sparsity(as.matrix(mnist.dat[,2:ncol(mnist.dat)]))
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
# V1 = label
# V2 = ink_cost scaled
ink_dataset <- as.data.frame(cbind(as.factor(mnist.dat$label), ink_scaled)) 
ink_model <- multinom(V1 ~ V2, ink_dataset)
ink_prediction <- predict(ink_model, ink_dataset)
ink_confusion_matrix <- table(ink_dataset$V1, ink_prediction)

## Third question: sparsity
sparse_values <- c()
for(i in 1:nrow(mnist.dat)){
  sparse_values[i] <- sparsity(as.matrix(mnist.dat[i,2:ncol(mnist.dat)]))
}
sparse_scaled <- scale(sparse_values)
sparse_dataset <- as.data.frame(cbind(as.factor(mnist.dat$label), sparse_scaled))
sparse_model <- multinom(V1 ~ V2, sparse_dataset)
sparse_prediction <- predict(sparse_model, sparse_dataset)
sparse_confusion_matrix <- table(sparse_dataset$V1, sparse_prediction)

## Fourth question: density (ink_cost) + sparsity  
ink_sparsity_dataset <- as.data.frame(cbind(as.factor(mnist.dat$label), ink_scaled, sparse_scaled))  
ink_sparsity_model <- multinom(V1 ~ ., ink_sparsity_dataset)  
ink_sparsity_prediction <- predict(ink_sparsity_model, ink_sparsity_dataset)
ink_sparsity_confusion_matrix <- table(ink_sparsity_dataset$V1, ink_sparsity_prediction)

#Fifth quesitons
set.seed(101)
sample <- sample.int(n = nrow(mnist.dat), size = 5000, replace = F)


training_data <- mnist.dat[sample,]
test_data <- mnist.dat[-sample,]
x <- training_data[,2:ncol(training_data)]
y <- as.numeric(training_data[, 1])

#Cross Validation Logistic Regression
cv_log_reg <- cv.glmnet(as.matrix(x), y, family = "multinomial")

#Cross validation SVM
cv_svm <- tune.svm(x,y)
summary(cv_svm)
plot(cv_svm)