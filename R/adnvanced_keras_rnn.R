# dir.create("~/Data/jena_climate", recursive = TRUE)
# download.file(
#   "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
#   "~/Data/jena_climate/jena_climate_2009_2016.csv.zip"
# )
# unzip(
#   "~/Data/jena_climate/jena_climate_2009_2016.csv.zip",
#   exdir = "~/Data/jena_climate"
# )

library(tibble)
library(tictoc)
library(keras)

data_dir <- "~/Download/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read.csv(fname, check.names = FALSE)

glimpse(data)

library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()

# preprocessing ####
data <- data.matrix(data[,-1])

train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), lookback/step, dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]] - 1, length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples, targets)
  }
}

lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps <- (300000 - 200001 - lookback) / batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

# non-machine learning baseline ####
evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

tic("calculating naive loss...")
naive_loss <- evaluate_naive_method()
naive_time <- toc()

# basic machine learning baseline ####
basic_model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

compile(basic_model, optimizer = optimizer_rmsprop(), loss = "mae")

tic("fitting basic model...")
basic_history <- fit_generator(basic_model, train_gen, steps_per_epoch = 500, 
                         epochs = 20, validation_data = val_gen, validation_steps = val_steps)
basic_time <- toc()

plot(basic_history)

# rnn baseline ####
gru_model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

compile(gru_model, optimizer = optimizer_rmsprop(), loss = "mae")

tic("fitting gru model...")
gru_history <- fit_generator(gru_model, train_gen, steps_per_epoch = 500, 
                             epochs = 20, validation_data = val_gen, validation_steps = val_steps)
gru_time <- toc()

plot(gru_history) # overfitting here

# combatting overfitting ####
gru_dropout_model <- keras_model_sequential() %>% 
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

compile(gru_dropout_model, optimizer = optimizer_rmsprop(), loss = "mae")

tic("fitting gru model with dropout...")
gru_dropout_history <- fit_generator(gru_dropout_model, train_gen, steps_per_epoch = 500, 
                         epochs = 40, validation_data = val_gen, validation_steps = val_steps)
gru_dropout_time <- toc()

plot(gru_dropout_history)

# stacking rnn layers ####
stacked_model <- keras_model_sequential() %>% 
  layer_gru(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)

compile(stacked_model, optimizer = optimizer_rmsprop(), loss = "mae")

tic("fitting stacked gru model...")
stacked_history <- fit_generator(stacked_model, train_gen, steps_per_epoch = 500, 
                                 epochs = 40, validation_data = val_gen, 
                                 validation_steps = val_steps)
stacked_time <- toc()

plot(stacked_history)
