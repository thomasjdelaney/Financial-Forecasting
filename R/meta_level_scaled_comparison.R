library(tictoc)
library(keras)
library(ggplot2)
library(M4comp2018)
library(forecast)
library(M4metalearning)

set.seed(1312)
yearly_M4 <- M4[which(sapply(M4, function(series)series$period == "Yearly"))]
yearly_forecast_horizon_length <- 6
indices <- sample(length(yearly_M4))
num_train_series <- 16000
num_val_series <- 3500
num_test_series <- 3500
yearly_test <- yearly_M4[(num_train_series+num_val_series+1):(num_train_series+num_val_series+num_test_series)]
yearly_fin_test <- yearly_test[which(sapply(yearly_test, function(series)series$type == "Finance"))]

# applying the meta learning model ####
meta_learning_model <- readRDS("./rds/M4_financial_xgboost_model.rds")

yearly_fin_test <- temp_holdout(yearly_fin_test)
yearly_fin_test <- THA_features(yearly_fin_test)
yearly_fin_test_data <- create_feat_classif_problem(yearly_fin_test)
meta_preds <- predict_selection_ensemble(meta_learning_model, yearly_fin_test_data$data)
meta_test <- ensemble_forecast(meta_preds, yearly_fin_test)

# applying the RNN model ####