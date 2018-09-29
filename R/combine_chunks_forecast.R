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

# loading the training set and creating the test set ####
set.seed(1202)
indices <- sample(num_fin_time_series)

#train
train_inds <- indices[10001:20000]
train <- fin_series[train_inds]
train <- temp_holdout(train)
tic("Foreacsting with statistical methods...")
train <- calc_forecasts(train, forec_methods(), n.cores = 6)
is_error <- sapply(train, function(series)length(names(series)) < 7)
train[is_error] <- NULL
train <- calc_errors(train)
train <- THA_features(train, n.cores = 6)
toc()

chunk1 <- readRDS("./rds/train_chunk_1.rds")
chunk2 <- readRDS("./rds/train_chunk_2.rds")
chunk3 <- readRDS("./rds/train_chunk_3.rds")
all_series <- c(chunk1, chunk2, chunk3)
rm(chunk1);rm(chunk2);rm(chunk3)
isfail <- sapply(all_series, function(sentry) all(is.na(sentry$errors)))

train <- c(all_series[which(!isfail)], train)
train_data <- create_feat_classif_problem(train)

#test
test_inds <- indices[20001:24534]
test <- fin_series[test_inds]
test <- c(test, readRDS("./rds/xlsx_file_data.rds"))
test <- temp_holdout(test)

# hyperparameter search ####
xgb_params <- hyperparameter_search(train, n.cores = 6, n_iter = 100)
best_xgb_params <- as.list(xgb_params$Best_Par)
best_xgb_params[["nthread"]] <- 6
best_xgb_params[["silent"]] <- 1
best_xgb_params[["objective"]] <- error_softmax_obj
best_xgb_params[["num_class"]] <- ncol(train_data$errors)

# training the model
set.seed(1345)
meta_model <- train_selection_ensemble(train_data$data, train_data$errors, best_xgb_params)

# extracting the features for the test set ####
test <- calc_forecasts(test, forec_methods(), n.cores = 6)
test <- THA_features(test, n.cores = 6)
test_data <- create_feat_classif_problem(test)

# forecasting on the test data ####
preds <- predict_selection_ensemble(meta_model, test_data$data)
test <- ensemble_forecast(preds, test)

summary <- summary_performance(preds, dataset = test)

saveRDS(train, "./rds/M4_financial_train.rds")
saveRDS(test, "./rds/augmented_M4_financial_test.rds")
saveRDS(xgb_params, "./rds/M4_financial_xgboost_params.rds")
saveRDS(meta_model, "./rds/M4_finanial_xgboost_model.rds")