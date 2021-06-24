mnist <- dataset_mnist()

x_train <- array_reshape(mnist$train$x, c(60000, 784))
x_train <- x_train / 255

y_train <- to_categorical(mnist$train$y)


x_test <- array_reshape(mnist$test$x, c(10000, 784))
x_test <- x_test / 255

y_test <- to_categorical(mnist$test$y)

model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, input_shape = c(784)) %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(), metrics = c("accuracy"))

history <- model %>% fit(x_train, y_train, batch_size = 128, epochs = 12, verbose = 1, validation_split = 0.2)

score <- model %>% evaluate(x_test, y_test, verbose = 0)

model2 <- keras_model_sequential()
model2 %>%
  layer_dense(units = 256, input_shape = c(784), activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model2 %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(), metrics = c("accuracy"))
history2 <- model2 %>% fit(x_train, y_train, batch_size = 128, epochs = 12, verbose = 1, validation_split = 0.2)

score2 <- model2 %>% evaluate(x_test, y_test, verbose = 0)

x_train <- array_reshape(mnist$train$x, c(60000,28,28, 1)) / 255
x_test <- array_reshape(mnist$test$x, c(10000, 28, 28, 1)) / 255

model3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")
model3 %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_adadelta(), metrics = c("accuracy"))  
history3 <- model3 %>% fit(x_train, y_train, batch_size = 128, epochs = 6, verbose = 1, validation_split = 0.2)
score3 <- model3 %>% evaluate(x_test, y_test, verbose = 0)

model4 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")
model4 %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_adadelta(), metrics = c("accuracy"))  
history4 <- model4 %>% fit(x_train, y_train, batch_size = 128, epochs = 6, verbose = 1, validation_split = 0.2)
score4 <- model4 %>% evaluate(x_test, y_test, verbose = 0)
 
cifar10 <- dataset_cifar10()
x_train_cifar <- cifar10$train$x / 255
x_test_cifar <- cifar10$test$x / 255

y_train_cifar <- to_categorical(cifar10$train$y)
y_test_cifar <- to_categorical(cifar10$test$y)

model5 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(32,32, 3), padding = "same") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")
model5 %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6), metrics = c("accuracy"))
history5 <- model5 %>% fit(x_train_cifar, y_train_cifar, batch_size = 32, epochs = 20, verbose = 1, validation_data = list(x_test_cifar, y_test_cifar), shuffle = TRUE)
