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

## Fifth question
{
  random_sample <- sort(sample(1:42000, 5000, replace=FALSE))
  training_set <- mnist.dat[random_sample,]
  testing_set <- mnist.dat[-random_sample,]
  
  training_set_14 <- data.frame()
  for (i in 1:nrow(training_set)){
    training_set_14 <- rbind(training_set_14, down_sample_image(as.matrix(training_set[i,2:ncol(training_set)]), 2))
  }
  training_set_14 <- cbind(training_set[,1], training_set_14)
  names(training_set_14)[1] <- "label"
  
  testing_set_14 <- data.frame()
  for (i in 1:nrow(testing_set)){
    testing_set_14 <- rbind(testing_set_14, down_sample_image(as.matrix(testing_set[i,2:ncol(testing_set)]), 2))
  }
  testing_set_14 <- cbind(testing_set[,1], testing_set_14)
  names(testing_set_14)[1] <- "label"
  
  ## 28 x 28
  {
    # 10-fold cross-validation -> multinomial model
    set.seed(42)
    regularized_multinomial_model <- cv.glmnet(x = as.matrix(training_set[,2:ncol(training_set)]), y = training_set[,1], family="multinomial")
    # 5-fold cross-validation -> multinomial model
    down_multinomial_model_5f <- cv.glmnet(x = as.matrix(training_set[,2:ncol(training_set)]), y = training_set[,1], nfolds = 5, family="multinomial")
    
    result_10f <- predict(regularized_multinomial_model, newx=as.matrix(testing_set[,2:ncol(testing_set)]), s="lambda.min", type="class")
    result_5f <- predict(down_multinomial_model_5f, newx=as.matrix(testing_set[,2:ncol(testing_set)]), s="lambda.min", type="class")
    
      
    confusion_matrix_10f <- ftable(testing_set[,1], result_10f)
    confusion_matrix_5f <- ftable(testing_set[,1], result_5f)
    
    acc_10f <- sum(diag(confusion_matrix_10f))/sum(confusion_matrix_10f)
    acc_5f <- sum(diag(confusion_matrix_5f))/sum(confusion_matrix_5f)

    # 10-fold cross-validation -> support vector machine model
    svm_model_tune <- tune.svm(as.factor(label) ~ ., data = training_set)
    svm_model <- svm(as.factor(label) ~ ., data = training_set, cross = 10)
    # 5-fold cross-validation -> support vector machine model
    svm_model_5f <- svm(as.factor(label) ~ ., data = training_set, cross = 5)
  
    # Neural network
    nn_model_tune <- tune.nnet(as.factor(label) ~ ., data = training_set, size = 5, MaxNWts = 20000)
    nn_model <- nnet(as.factor(label) ~ ., data = training_set, size = 5, MaxNWts = 5000)
  }
    
  ## 14 x 14
  {
    # 10-fold cross-validation -> multinomial model
    set.seed(42)
    regularized_multinomial_model_14 <- cv.glmnet(x = as.matrix(training_set_14[,2:ncol(training_set)]), y = training_set_14[,1], family="multinomial")
    # 5-fold cross-validation -> multinomial model
    down_multinomial_model_5f_14 <- cv.glmnet(x = as.matrix(training_set_14[,2:ncol(training_set)]), y = training_set_14[,1], nfolds = 5, family="multinomial")
    
    result_10f <- predict(regularized_multinomial_model_14, newx=as.matrix(testing_set_14[,2:ncol(testing_set_14)]), s="lambda.min", type="class")
    result_5f <- predict(down_multinomial_model_5f_14, newx=as.matrix(testing_set_14[,2:ncol(testing_set_14)]), s="lambda.min", type="class")
    
    
    confusion_matrix_10f_14 <- ftable(testing_set[,1], result_10f)
    confusion_matrix_5f_14 <- ftable(testing_set[,1], result_5f)
    
    acc_10f_14 <- sum(diag(confusion_matrix_10f_14))/sum(confusion_matrix_10f_14)
    acc_5f_14 <- sum(diag(confusion_matrix_5f_14))/sum(confusion_matrix_5f_14)
    
    # Delete columns with 0
    columns_delete <- c()
    for(i in 2:ncol(training_set_14)){
      if(!any(training_set_14[,i])){
        columns_delete <- c(columns_delete, i)
      }
    }
    gamma_range <- logspace(-9, 3, n = 20)
    # 10-fold cross-validation -> support vector machine model
    down_svm_model_tune <- tune.svm(as.factor(label) ~ ., data = training_set_14, cost = 1:10, gamma = gamma_range, 
                                    degree = c(1:10), epsilon = logspace(-1, 0.5, n=3))
    down_svm_model <- svm(as.factor(label) ~ ., data = training_set_14, cross = 10)
  
    # Neural network
    down_nn_model_tune <- tune.nnet(as.factor(label) ~ ., data = training_set_14, size = logspace(0, 5, n = 6), 
                                    decay = , MaxNWts = 5000)
    down_nn_model <- nnet(as.factor(label) ~ ., data = training_set_14, size = 5, MaxNWts = 5000)
  }
}
