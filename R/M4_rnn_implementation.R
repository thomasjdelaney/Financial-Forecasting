library(tictoc)
library(keras)
library(ggplot2)
library(M4comp2018)
library(forecast)

set.seed(1312)
yearly_M4 <- M4[which(sapply(M4, function(series)series$period == "Yearly"))]
yearly_forecast_horizon_length <- 6
indices <- sample(length(yearly_M4))
num_train_series <- 16000
num_val_series <- 3500
num_test_series <- 3500
yearly_train <- yearly_M4[1:num_train_series]
yearly_val <- yearly_M4[(num_train_series+1):(num_train_series+num_val_series)]
yearly_test <- yearly_M4[(num_val_series+1):(num_val_series+num_test_series)]

# transforming training data ####
num_points_to_use <- floor(median(sapply(yearly_train, function(series)length(series$x))))

combineSeriesHoldoutM4 <- function(M4_series) {
  comb <- ts.union(M4_series$x, M4_series$xx)
  pmin(comb[,1], comb[,2], na.rm = TRUE)
}

data_gen <- function(dataset) {
  num_series <- length(dataset)
  samples <- array(0, dim = c(num_train_series, num_points_to_use, 4))
  values <- array(0, dim = c(num_train_series, yearly_forecast_horizon_length))
  for (i in 1:num_series){
    series <- dataset[[i]]
    series_to_use <- tail(series$x, num_points_to_use)
    num_points <- length(series_to_use)
    ets_model <- ets(series_to_use, model = "MMN")
    level <- tail(ets_model$states[,1], num_points)
    trend <- tail(ets_model$states[,2], num_points)
    residuals <- ets_model$residuals
    if (num_points < num_points_to_use) { # padding wtih zeros
      series_to_use <- c(0*1:(num_points_to_use - num_points), series_to_use)
      level <- c(0*1:(num_points_to_use - num_points), level)
      trend <- c(1+0*1:(num_points_to_use - num_points), trend) # padding with 1s here.
      residuals <- c(0*1:(num_points_to_use - num_points), residuals)
    }
    samples[i,,1] <- series_to_use
    samples[i,,2] <- level
    samples[i,,3] <- trend
    samples[i,,4] <- residuals
    values[i,] <- series$xx
  }
  return(list(samples, values))
}

c(train_samples, train_values) %<-% data_gen(yearly_train)
train_means <- apply(train_samples, 3, mean)
train_sds <- apply(train_samples, 3, sd)

# scaling data ####
for (i in 1:num_train_series){
  train_samples[i,,] <- scale(train_samples[i,,], center = train_means, scale = train_sds)
  train_values[i,] <- scale(train_values[i,], center = train_means[[1]], scale = train_sds[[1]])
}

c(val_samples, val_values) %<-% data_gen(yearly_val)
for (i in 1:num_val_series){
  val_samples[i,,] <- scale(val_samples[i,,], center = train_means, scale = train_sds)
  val_values[i,] <- scale(val_values[i,], center = train_means[[1]], scale = train_sds[[1]])
}

c(test_samples, test_values) %<-% data_gen(yearly_test)
for (i in 1:num_test_series){
  test_samples[i,,] <- scale(test_samples[i,,], center = train_means, scale = train_sds)
  test_values[i,] <- scale(test_values[i,], center = train_means[[1]], scale = train_sds[[1]])
}

# naive method ####
evaluate_naive_method <- function() {
  scaled_batch_maes <- array(0, dim = c(num_val_series))
  batch_maes <- array(0, dim = c(num_val_series))
  for (i in 1:num_val_series) {
    val_series <- val_samples[i,,1]
    val_value <- val_values[i,]
    rw_drift_model <- rwf(val_series, drift = TRUE, h = yearly_forecast_horizon_length)
    fc <- forecast(rw_drift_model, h = yearly_forecast_horizon_length)$mean
    scaled_batch_maes[[i]] <- mean(abs(fc - val_value))
    val_value <- (val_value * train_sds[[1]]) + train_means[[1]]
    fc <- (fc * train_sds[[1]]) + train_means[[1]]
    batch_maes[[i]] <- mean(abs(fc - val_value))
  }
  print(mean(scaled_batch_maes))
  print(mean(batch_maes))
}

tic("calculating naive loss...")
naive_loss <- evaluate_naive_method()
naive_time <- toc()

# simple gru network ####
gru_model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, 4), return_sequences = TRUE, dropout = 0.1, recurrent_dropout = 0.5) %>%
  layer_gru(units = 64, activation = "relu", dropout = 0.1, recurrent_dropout = 0.5) %>%
  layer_dense(units = yearly_forecast_horizon_length)

compile(gru_model, optimizer = optimizer_rmsprop(), loss = "mae")

tic("fitting gru model...")
gru_history <- fit(gru_model, x = train_samples, y = train_values, steps_per_epoch = 500, epochs = 40, validation_data = list(val_samples, val_values), validation_steps = 100, view_metrics = TRUE)
gru_time <- toc()

evaluate_model <- function(model, samples, values){
  num_samples <- dim(samples)[[1]]
  preds <- predict(model, x = samples)
  scaled_maes <- array(0, dim = c(num_samples))
  maes <- array(0, dim = c(num_samples))
  for (i in 1:num_samples) {
    value <- values[i,]
    pred <- preds[i,]
    scaled_maes[[i]] <- mean(abs(pred - value))
    value <- (value * train_sds[[1]]) + train_means[[1]]
    pred <- (pred * train_sds[[1]]) + train_means[[1]]
    maes[[i]] <- mean(abs(pred - value))
  }
  print(mean(scaled_maes))
  print(mean(maes))
}

evaluate_model(gru_model, val_samples, val_values)
plot(gru_history)