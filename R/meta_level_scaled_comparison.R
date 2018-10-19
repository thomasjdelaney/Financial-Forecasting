library(tictoc)
library(keras)
library(ggplot2)
library(M4comp2018)
library(forecast)
library(M4metalearning)
library(M4metaresults)

# loading forecasted test series ####
num_train_series <- 16000
num_val_series <- 3500
num_test_series <- 3500

yearly_submission <- submission_M4[which(sapply(submission_M4, function(series)series$period == "Yearly"))]
yearly_submission <- temp_holdout(yearly_submission)
quarterly_submission <- submission_M4[which(sapply(submission_M4, function(series)series$period == "Quarterly"))]
quarterly_submission <- temp_holdout(quarterly_submission)
yearly_test_submission <- yearly_submission[(16000+3500+1):(16000+3500+3500)]
quarterly_test_submission <- quarterly_submission[(17000+3500+1):(17000+3500+3500)]
yearly_meta_preds <- t(sapply(yearly_test_submission, function(series)series$y_hat))
quarterly_meta_preds <- t(sapply(quarterly_test_submission, function(series)series$y_hat))

# applying the RNN model ####
level_scaled_model <- load_model_hdf5("./h5/residual_gru_model.h5")

num_points_to_use <- 29
forecast_horizon_length <- 6

test_gen <- function(){ # can't use the values in the holdout set when calculating the level and alpha for test data
  num_series <- length(yearly_test_submission)
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

c(level_scaled_samples, level_series, alphas, test_values, num_series)  %<-% test_gen()

rnn_scaled_preds <- predict(level_scaled_model, x = level_scaled_samples)

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

rnn_preds <- array(0, dim = c(num_series, forecast_horizon_length))
level_preds <- array(0, dim = c(num_series, forecast_horizon_length))
for (i in 1:num_series){
  descaled <- getPredictionFromLevelParams(rnn_scaled_preds[i,], level_series[i,], alphas[[i]])
  rnn_preds[i,] <- descaled[[1]]
  level_preds[i,] <- descaled[[2]]
}
rnn_residuals <- rnn_preds - test_values
meta_residuals <- yearly_meta_preds - test_values