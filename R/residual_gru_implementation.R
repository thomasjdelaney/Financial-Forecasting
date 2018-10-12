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
yearly_test <- yearly_M4[(num_train_series+num_val_series+1):(num_train_series+num_val_series+num_test_series)]

# transforming training data ####
num_points_to_use <- floor(median(sapply(yearly_train, function(series)length(series$x))))

combineSeriesHoldout <- function(M4_series) {
  comb <- ts.union(M4_series$x, M4_series$xx)
  pmin(comb[,1], comb[,2], na.rm = TRUE)
}

data_gen <- function(dataset) {
  num_series <- length(dataset)
  samples <- array(0, dim = c(num_series, num_points_to_use, 1))
  values <- array(0, dim = c(num_series, yearly_forecast_horizon_length))
  scaled_values <- array(0, dim = c(num_series, yearly_forecast_horizon_length))
  level_series <- array(0, dim = c(num_series, num_points_to_use))
  alphas <- array(0, dim = c(num_series))
  for (i in 1:num_series){
    series <- dataset[[i]]
    series_sample <- tail(series$x, num_points_to_use)
    series_value <- series$xx
    sample_points <- length(series_sample)
    ets_model <- ets(combineSeriesHoldout(series), model = "MNN")
    level_sample <- ets_model$states[1:sample_points,1]
    level_value <- ets_model$states[(sample_points+1):(sample_points+1+yearly_forecast_horizon_length),1]
    series_sample <- head(ets_model$residuals, sample_points)
    series_value <- tail(ets_model$residuals, yearly_forecast_horizon_length)
    if (sample_points < num_points_to_use) {# padding with zeros
      series_sample <- c(0*1:(num_points_to_use - sample_points), series_sample)
      level_sample <- c(head(level_sample, 1) + 0*1:(num_points_to_use - sample_points), level_sample)
    }
    samples[i,,1] <- series_sample
    level_series[i,] <- level_sample
    values[i,] <- series$xx
    scaled_values[i,] <- series_value
    alphas[[i]] <- ets_model$par[[1]]
  }
  return(list(samples, values, level_series, scaled_values, alphas))
}

test_gen <- function(){ # can't use the values in the holdout set when calculating the level and alpha for test data
  num_series <- length(yearly_test)
  samples <- array(0, dim = c(num_series, num_points_to_use, 1))
  level_series <- array(0, dim = c(num_series, num_points_to_use))
  alphas <- array(0, dim = c(num_series))
  values <- array(0, dim = c(num_series, yearly_forecast_horizon_length))
  for (i in 1:num_series){
    series <- yearly_test[[i]]
    series_sample <- tail(series$x, num_points_to_use)
    sample_points <- length(series_sample)
    ets_model <- ets(series_sample, model = "MNN")
    level_sample <- ets_model$states[1:sample_points,1]
    series_sample <- ets_model$residuals
    if (sample_points < num_points_to_use) {# padding with zeros
      series_sample <- c(0*1:(num_points_to_use - sample_points), series_sample)
      level_sample <- c(head(level_sample, 1) + 0*1:(num_points_to_use - sample_points), level_sample)
    }
    samples[i,,1] <- series_sample
    level_series[i,] <- level_sample
    alphas[[i]] <- ets_model$par[[1]]
    values[i,] <- series$xx
  }
  return(list(samples, level_series, alphas, values))
}

c(train_samples, train_values, train_levels, train_scaled_values, train_alphas) %<-% data_gen(yearly_train)
c(val_samples, val_values, val_levels, val_scaled_values, val_alphas) %<-% data_gen(yearly_val)
c(test_samples, test_levels, test_alphas, test_values) %<-% test_gen()

# naive method ####
evaluate_naive_method <- function() {
  num_series <- length(yearly_val)
  batch_maes <- array(0, dim = c(num_series))
  for (i in 1:num_series) {
    val_series <- yearly_val[[i]]
    val_samples <- val_series$x
    val_value <- val_series$xx
    rw_drift_model <- rwf(val_samples, drift = TRUE, h = yearly_forecast_horizon_length)
    fc <- forecast(rw_drift_model, h = yearly_forecast_horizon_length)$mean
    batch_maes[[i]] <- mean(abs(fc - val_value))
  }
  print(mean(batch_maes))
}

tic("calculating naive loss...")
naive_loss <- evaluate_naive_method()
naive_time <- toc()

# simple gru network ####
gru_model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, 1), return_sequences = TRUE, dropout = 0.1, recurrent_dropout = 0.5) %>%
  layer_gru(units = 64, activation = "relu", dropout = 0.1, recurrent_dropout = 0.5) %>%
  layer_dense(units = yearly_forecast_horizon_length)

compile(gru_model, optimizer = optimizer_rmsprop(), loss = "mae")

tic("fitting gru model...")
gru_history <- fit(gru_model, x = train_samples, y = train_scaled_values, steps_per_epoch = 500, epochs = 40, validation_data = list(val_samples, val_scaled_values), validation_steps = 100, view_metrics = TRUE)
gru_time <- toc()

getPredictionFromLevelParams <- function(model_predictions, level_series, alpha) {
  num_predictions <- length(model_predictions)
  level_preds <- c(tail(level_series, 1), array(0, dim = c(num_predictions))) # prepending last level value
  series_preds <- array(0, dim = c(num_predictions))
  for (i in 1:num_predictions){
    series_preds[[i]] <- level_preds[[i]]*(1 + model_predictions[[i]])
    level_preds[[i+1]] <- level_preds[[i]]*(1 + alpha*model_predictions[[i]])
  }
  return(list(series_preds, tail(level_preds, num_predictions)))
}

preds <- predict(gru_model, x = test_samples)
actual_preds <- array(0, dim = c(num_test_series, yearly_forecast_horizon_length))
level_preds <- array(0, dim = c(num_test_series, yearly_forecast_horizon_length))
for (i in 1:num_test_series){
  descaled <- getPredictionFromLevelParams(preds[i,], test_levels[i,], test_alphas[[i]])
  actual_preds[i,] <- descaled[[1]]
  level_preds[i,] <- descaled[[2]]
}
print(mean(abs(actual_preds - test_values)))