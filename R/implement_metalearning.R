library(rBayesianOptimization)
library(M4metalearning)
library(M4comp2018)

# Isolating the Financial Time Series ####

isfin <- sapply(M4, function(time_series) time_series$type == "Finance")
fin_inds <- which(isfin)
num_fin_time_series <- length(fin_inds)
fin_series <- M4[fin_inds]

# making some graphs ####
source("R/data_exploration.R")

# creating the training and test sets ####
indices <- sample(num_fin_time_series)
train_inds <- indices[1:20] # 4/5
test_inds <- indices[21:25]
train <- fin_series[train_inds]
test <- fin_series[test_inds]
train <- temp_holdout(train)
test <- temp_holdout(test)

train <- calc_forecasts(train, forec_methods(), n.cores=3)
train <- calc_errors(train)
train <- THA_features(train, n.cores=3)
train_data <- create_feat_classif_problem(train)

# finding the best hyperparameters ####
xgb_params <- hyperparameter_search(train, n.cores=3, n_iter=20)
best_xgb_params = as.list(xgb_params$Best_Par)
best_xgb_params[["nthread"]] <- 3
best_xgb_params[["silent"]] <- 1
best_xgb_params[["objective"]] <- error_softmax_obj
best_xgb_params[["num_class"]] <- ncol(train_data$errors)

# training the model and forecasting ####
set.seed(1345) 
meta_model <- train_selection_ensemble(train_data$data, train_data$errors, best_xgb_params)

test <- calc_forecasts(test, forec_methods(), n.cores=3)
test <- THA_features(test, n.cores=3)
test_data <- create_feat_classif_problem(test)
preds <- predict_selection_ensemble(meta_model, test_data$data)
test <- ensemble_forecast(preds, test)