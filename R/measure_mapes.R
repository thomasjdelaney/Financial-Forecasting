library(M4comp2018)
library(TSPred)

plotForecastVsActual <- function(series){
  forecast_data <- tail(c(series$x, series$y_hat), 5*series$h)
  actual_data <- tail(c(series$x, series$xx), 5*series$h)
  y_limits <- c(min(c(forecast_data, actual_data)), max(c(forecast_data, actual_data)))
  series_title <- paste(series$st, "Price")
  plot(forecast_data, type="l", col="blue", ylim = y_limits, ylab = series_title)
  lines(actual_data, type="l", col="black")
}

calcMDapes <- function(series){
  series_mape <- median(abs((series$xx - series$y_hat)/series$xx))
  rw_drift_mape <- median(abs((series$xx - series$ff["rw_drift_forec",])/series$xx))
  arima_mape <- median(abs((series$xx - series$ff["auto_arima_forec",])/series$xx))
  ets_mape <- median(abs((series$xx - series$ff["ets_forec",])/series$xx))
  nn_mape <- median(abs((series$xx - series$ff["nnetar_forec",])/series$xx))
  tbats_mape <- median(abs((series$xx - series$ff["tbats_forec",])/series$xx))
  stlm_mape <- median(abs((series$xx - series$ff["stlm_ar_forec",])/series$xx))
  thetaf_mape <- median(abs((series$xx - series$ff["thetaf_forec",])/series$xx))
  naive_mape <- median(abs((series$xx - series$ff["naive_forec",])/series$xx))
  snaive_mape <- median(abs((series$xx - series$ff["snaive_forec",])/series$xx))
  series_frame <- data.frame(series_name = series$st, 
                             meta_mape = series_mape, 
                             rw_drift_mape = rw_drift_mape, 
                             arima_mape = arima_mape,
                             ets_mape = ets_mape,
                             nn_mape = nn_mape,
                             tbats_mape = tbats_mape,
                             stlm_mape = stlm_mape,
                             thetaf_mape = thetaf_mape,
                             naive_mape = naive_mape,
                             snaive_mape = snaive_mape)
  return(series_frame)
}

# Setting up globals ####
isfin <- sapply(M4, function(time_series) time_series$type == "Finance")
fin_inds <- which(isfin)
num_fin_time_series <- length(fin_inds)
set.seed(1202)
indices <- sample(num_fin_time_series)
test_inds <- indices[20001:24534]
test <- readRDS("./rds/augmented_M4_financial_test_daily.rds")

length_cr_test <- length(test) - length(test_inds)
cr_test <- tail(test, length_cr_test)
M4_test <- head(test, length(test_inds))

# measuring errors on CheckRisk dataset ####
cr_error_table <- data.frame(series_name = character(), 
                             meta_mape = double(), 
                             rw_drift_mape = double(), 
                             arima_mape = double(),
                             ets_mape = double(),
                             nn_mape = double(),
                             tbats_mape = double(),
                             stlm_mape = double(),
                             thetaf_mape = double(),
                             naive_mape = double(),
                             snaive_mape = double())
for (i in 1:length_cr_test){
  series <- cr_test[[i]]
  # plotForecastVsActual(series)
  series_frame <- calcMDapes(series)
  cr_error_table <- rbind(cr_error_table, series_frame)
}

# measuring errors on all the M4 test ####
M4_error_table <- data.frame(series_name = character(), 
                             meta_mape = double(), 
                             rw_drift_mape = double(), 
                             arima_mape = double(),
                             ets_mape = double(),
                             nn_mape = double(),
                             tbats_mape = double(),
                             stlm_mape = double(),
                             thetaf_mape = double(),
                             naive_mape = double(),
                             snaive_mape = double())
for (i in 1:length(M4_test)){
  series <- M4_test[[i]]
  series_frame <- calcMDapes(series)
  M4_error_table <- rbind(M4_error_table, series_frame)
}

# Measuring errors on M4 test dataset subset of weekly data ####
M4_test_weekly <- M4_test[which(sapply(M4_test, function(series)(series$period == "Weekly")||(series$period == "Daily")))]
length_M4_weekly <- length(M4_test_weekly)
M4_error_table_weekly <- data.frame(series_name = character(), 
                             meta_mape = double(), 
                             rw_drift_mape = double(), 
                             arima_mape = double(),
                             ets_mape = double(),
                             nn_mape = double(),
                             tbats_mape = double(),
                             stlm_mape = double(),
                             thetaf_mape = double(),
                             naive_mape = double(),
                             snaive_mape = double())
for (i in 1:length_M4_weekly){
  series <- M4_test_weekly[[i]]
  # plotForecastVsActual(series)
  series_frame <- calcMapes(series)
  M4_error_table_weekly <- rbind(M4_error_table_weekly, series_frame)
}

# Measuring errors on M4 test dataset subset of daily data ####
M4_test_daily <- M4_test[which(sapply(M4_test, function(series)series$period == "Daily"))]
length_M4_daily <- length(M4_test_daily)
M4_error_table_daily <- data.frame(series_name = character(), 
                                   meta_mape = double(), 
                                   rw_drift_mape = double(), 
                                   arima_mape = double(),
                                   ets_mape = double(),
                                   nn_mape = double(),
                                   tbats_mape = double(),
                                   stlm_mape = double(),
                                   thetaf_mape = double(),
                                   naive_mape = double(),
                                   snaive_mape = double())
for (i in 1:length_M4_daily){
  series <- M4_test_daily[[i]]
  # plotForecastVsActual(series)
  series_frame <- calcMapes(series)
  M4_error_table_daily <- rbind(M4_error_table_daily, series_frame)
}

write.csv(cr_error_table, file = "./csv/finance_model_checkRisk_data_forecasting_errors.csv")
write.csv(M4_error_table, file = "./csv/finance_model_M4_weekly_data_forecasting_errors.csv")
write.csv(M4_error_table_daily, file = "./csv/finance_model_M4_daily_data_forecasting_errors.csv")