library(rBayesianOptimization)
library(M4metalearning)
library(M4comp2018)
library(rbenchmark)
library(tictoc)

# Isolating the Financial Time Series ####

isfin <- sapply(M4, function(time_series) time_series$type == "Finance")
fin_inds <- which(isfin)
num_fin_time_series <- length(fin_inds)
fin_series <- M4[fin_inds]

# creating the training and test sets ####
set.seed(1202)
indices <- sample(num_fin_time_series)
train_inds <- indices[1:30] 
test_inds <- indices[31:40]
train <- fin_series[train_inds]
test <- fin_series[test_inds]
train <- temp_holdout(train)
test <- temp_holdout(test)

tic("Benchmarking: num_train_series = 10000, num_iter = 1")
b10000 <- benchmark("train_forecasts_features" = {
  message(paste(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"),"forecasting the training set using the basic models..."))
  train <- calc_forecasts(train, forec_methods(), n.cores = 6)
  train <- calc_errors(train)
  train <- THA_features(train, n.cores = 6)
  train_data <- create_feat_classif_problem(train)
},
"hyperparameter_search" = {
  message(paste(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"),"searching the hyperparameter space..."))
  xgb_params <- hyperparameter_search(train, n.cores = 6, n_iter = 1)
  best_xgb_params = as.list(xgb_params$Best_Par)
  best_xgb_params[["nthread"]] <- 6
  best_xgb_params[["silent"]] <- 1
  best_xgb_params[["objective"]] <- error_softmax_obj
  best_xgb_params[["num_class"]] <- ncol(train_data$errors)
},
"train_model" = {
  set.seed(1345)
  message(paste(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"),"training the meta-learning model..."))
  meta_model <- train_selection_ensemble(train_data$data, train_data$errors, best_xgb_params)
},
"test_forecasts_features" = {
  message(paste(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"),"forecasting the test set using the basic models..."))
  test <- calc_forecasts(test, forec_methods(), n.cores = 6)
  test <- THA_features(test, n.cores = 6)
  test_data <- create_feat_classif_problem(test)
},
"forecast" = {
  message(paste(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"),"forecasting using the meta-learning model..."))
  preds <- predict_selection_ensemble(meta_model, test_data$data)
},
replications = 0,
columns = c("test", "replications", "elapsed",
            "relative", "user.self", "sys.self"))
toc()
saveRDS(train, "./rds/train.rds")
saveRDS(train_data, "./rds/train_data.rds")
saveRDS(best_xgb_params, "./rds/best_xgb_params.rds")
saveRDS(meta_model, "./rds/meta_model.rds")
saveRDS(test, "./rds/test.rds")
saveRDS(test_data, "./rds/test_data.rds")
saveRDS(preds, "./rds/preds.rds")