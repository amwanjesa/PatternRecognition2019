library(OpenImageR)
library(coop)
library(nnet)
library(glmnet)
library(e1071)
library(pracma)

mnist.dat <- read.csv("mnist.csv")

## First question
{
  sparse <- sparsity(as.matrix(mnist.dat[,2:ncol(mnist.dat)]))
  means <- c()
  means <- colMeans(mnist.dat[,2:ncol(mnist.dat)])

  imageShow(matrix(as.numeric(means),nrow=28,ncol=28,byrow=T))
  label_counts <- summary(as.factor(mnist.dat[,1]))
  majority_label <- label_counts[order(label_counts, decreasing = TRUE)]
  error_majority_label <- (nrow(mnist.dat) - as.numeric(majority_label[1])) / nrow(mnist.dat)
}
  
## Second question: density (ink_cost)
{
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
}
  
## Third question: sparsity
{
  sparse_values <- c()
  for(i in 1:nrow(mnist.dat)){
    sparse_values[i] <- sparsity(as.matrix(mnist.dat[i,2:ncol(mnist.dat)]))
  }
  sparse_scaled <- scale(sparse_values)
  sparse_dataset <- as.data.frame(cbind(as.factor(mnist.dat$label), sparse_scaled))
  sparse_model <- multinom(V1 ~ V2, sparse_dataset)
  sparse_prediction <- predict(sparse_model, sparse_dataset)
  sparse_confusion_matrix <- table(sparse_dataset$V1, sparse_prediction)
}
  
## Fourth question: density (ink_cost) + sparsity
{
  ink_sparsity_dataset <- as.data.frame(cbind(as.factor(mnist.dat$label), ink_scaled, sparse_scaled))  
  ink_sparsity_model <- multinom(V1 ~ ., ink_sparsity_dataset)  
  ink_sparsity_prediction <- predict(ink_sparsity_model, ink_sparsity_dataset)
  ink_sparsity_confusion_matrix <- table(ink_sparsity_dataset$V1, ink_sparsity_prediction)
}

## Fifth question
{
  ## Training and testing data
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
  }
  
  ## 28 x 28
  {
    # Multinomial model
    regularized_multinomial_model <- cv.glmnet(x = as.matrix(training_set[,2:ncol(training_set)]), y = training_set[,1])
    down_multinomial_model_5f <- cv.glmnet(x = as.matrix(training_set[,2:ncol(training_set)]), y = training_set[,1], nfolds = 5)

    # Support vector machine model
    svm_model_tune <- tune.svm(as.factor(label) ~ ., data = training_set)
    svm_model <- svm(as.factor(label) ~ ., data = training_set, cross = 10)
    
    # Neural network
    nn_model_tune <- tune.nnet(as.factor(label) ~ ., data = training_set, size = 5, MaxNWts = 20000)
    nn_model <- nnet(as.factor(label) ~ ., data = training_set, size = 5, MaxNWts = 5000)
  }
    
  ## 14 x 14
  {
    # Multinomial model
    down_multinomial_model <- cv.glmnet(x = as.matrix(training_set_14[,2:ncol(training_set_14)]), y = training_set_14[,1],
                                        alpha = c(0.001, 0.01, 1, 3, 5, 10), nlambda = c(10, 50, 100, 200, 500, 1000))
    
    # Delete columns with 0
    columns_delete <- c()
    for(i in 2:ncol(training_set_14)){
      if(!any(training_set_14[,i])){
        columns_delete <- c(columns_delete, i)
      }
    }
    
    # Rename data sets
    for(i in 2:ncol(training_set_14)){
      names(training_set_14)[i] <- i
      names(testing_set_14)[i] <- i
    }

    # Support vector machine
    down_svm_model_tune <- tune.svm(as.factor(label) ~ ., data = training_set_14, cost = 1:10, gamma = logspace(-9, 3, n = 5))
    down_svm_model <- svm(as.factor(label) ~ ., data = training_set_14[,-columns_delete], cost = 2, gamma = 1e-06)
  
    # Neural network
    down_nn_model_tune <- tune.nnet(as.factor(label) ~ ., data = training_set_14, size = c(1, 5, 10, 25), decay = c(0.1, 0.5, 1), MaxNWts = 50000)
    down_nn_model <- nnet(as.factor(label) ~ ., data = training_set_14, size = 10, MaxNWts = 5000)
  }
}
