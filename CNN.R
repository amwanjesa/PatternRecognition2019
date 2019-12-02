# Libraries
library(keras)

# Data
xt <- c()
for(i in 1:nrow(training_set)){
  xt <- c(xt, (matrix(as.numeric(training_set[i,-1]),nrow=28,ncol=28,byrow=T)))
}

xte <- c()
for(i in 1:nrow(testing_set)){
  xte <- c(xte, (matrix(as.numeric(testing_set[i,-1]),nrow=28,ncol=28,byrow=T)))
}

# Data reshape
x_train = xt
dim(x_train) = c(5000, 28, 28, 1)
x_test = xte
dim(x_test) = c(37000, 28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(training_set[,1], 10)
y_test = to_categorical(testing_set[,1], 10)

# Model definition
model = keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = 'relu', input_shape = c(28,28,1))%>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(loss = 'categorical_crossentropy',optimizer = optimizer_adam(),metrics = c('accuracy'))

# Training
history <-model %>% fit(x_train, y_train, batch_size = 64,epochs = 10, verbose = 1, validation_split = 0.2)

score <-model %>% evaluate(x_test, y_test,verbose = 0)
