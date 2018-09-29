library(readxl)
library(dplyr)

rm.emptycol <- function(dt) {
  idx <- apply(dt , MARGIN = 2, FUN = function (x) { !all(is.na(x)) } )
  return ( dt[,idx] )
}

cleanUp <- function(dt) {
  names(dt) <- c("Date" , "Index" ) # Rename Series
  dt <- dt[!is.na (dt$Date),] # Remove data where date is NA
  # Convert Date
  # check if it is excel date format - it probably is!!
  if (nchar(as.character(tail(dt$Date,1))) == 5) {
    dt$Date <- excel.Date (as.numeric(as.character(dt$Date)))
  } else {
    e <- tryCatch({ dt$Date <- as.Date (dt$Date) } ,
                  error = function(e) {
                    warning ( "Unrecognised date format trying excel format number format")
                    return ( -999 )
                  })
    if (first(e) == -999) {
      dt$Date <- excel.Date(as.numeric(as.character(dt$Date)))
    }
  }
  dt$Index <- as.numeric(dt$Index)
  dt <- dt[!is.na (dt$Date),] # Remove data where date is NA
  return(dt)
}


excel.Date <- function(numbers) {
  return(as.Date("1900/01/01") + numbers - 2)
}

getPeriodFromDays <- function(lag_days) {
  case_when(
    lag_days <= 6 ~ "Daily",
    (lag_days > 6) & (lag_days <= 20) ~ "Weekly",
    (lag_days > 20) & (lag_days <= 40) ~ "Monthly",
    (lag_days > 40) & (lag_days <= 100) ~ "Quarterly",
    (lag_days > 100) & (lag_days <= 360) ~ "Yearly",
    TRUE ~ "Unknown period"
  )
}

getForecastHorizonFromPeriod <- function(period) {
  case_when(
    period == "Daily" ~ 14,
    period == "Weekly" ~ 13,
    period == "Monthly" ~ 18,
    period == "Quarterly" ~ 8,
    period == "Yearly" ~ 6,
    TRUE ~ 0
  )
}

removeInitialLowFreqRecords <- function(period, max_date_diff, clean_series) {
  # NB: Only removes records from the start of the time series!
  threshold_date_diff <- case_when(
    period == "Daily" ~ 8,
    period == "Weekly" ~ 28,
    period == "Monthly" ~ 3,
    period == "Quarterly" ~ 4,
    period == "Yearly" ~ 5,
    TRUE ~ 0
  )
  if (max_date_diff >= threshold_date_diff) {
    last_low_freq_report_index <- tail(which(diff.Date(clean_series$Date) >= threshold_date_diff),1)
    message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"), " cutting out ", last_low_freq_report_index, " records...")
    clean_series <- tail(clean_series, -last_low_freq_report_index)
  }
  return(clean_series)
}

processSeries <- function(sheet, col_names, sheet_series, series_number, minimum_length) {
  # series number is just for indexing
  start_ind <- 2*series_number-1
  end_ind <- start_ind+1
  sheet_series[[series_number]]$st <- col_names[start_ind]
  clean_series <- cleanUp(sheet[col_names[start_ind:end_ind]])
  if (length(clean_series$Index) <= minimum_length){
    message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"), " length(", col_names[start_ind], ") <= ", minimum_length, "! Skipping.")
    sheet_series[[series_number]]$n <- 0
    return(sheet_series)
  }
  mean_date_diff <- mean(diff.Date(clean_series$Date))
  max_date_diff <- max(diff.Date(clean_series$Date))
  period <- getPeriodFromDays(mean_date_diff)
  clean_series <- removeInitialLowFreqRecords(period, max_date_diff, clean_series)  
  if (length(clean_series$Index) <= minimum_length){
    message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"), " length(", col_names[start_ind], ") <= ", minimum_length, "! Skipping.")
    sheet_series[[series_number]]$n <- 0
    return(sheet_series)
  }
  clean_series$Index <- as.ts(clean_series$Index)
  sheet_series[[series_number]]$mean_date_diff <- mean_date_diff
  sheet_series[[series_number]]$max_date_diff <- max_date_diff
  sheet_series[[series_number]]$period <- period
  h <- getForecastHorizonFromPeriod(sheet_series[[series_number]]$period)
  n_clean <- length(clean_series$Index)
  if ((5*h) >= n_clean){
    message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3 "), col_names[start_ind], " is short. Using custom horizon...")
    h <- ceiling(n_clean/5)
  }
  n <- n_clean - h
  sheet_series[[series_number]]$h <- h
  sheet_series[[series_number]]$n <- n
  sheet_series[[series_number]]$x <- head(clean_series$Index, n)
  sheet_series[[series_number]]$xx <- tail(clean_series$Index, h)
  sheet_series[[series_number]]$type <- "Finance"
  return(sheet_series)
}

ep_file <- "./data/xlsx/regions_sectors_daily.xlsx"
minimum_length <- 300
sheet_names <- list("Equity Regions", "US Sectors", "Fixed Income")
file_series <- list()
for (sheet_name in sheet_names) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3")," Processing equity_price.xlsx, sheet: ", sheet_name, "...")
  sheet <- rm.emptycol(read_excel(ep_file, sheet=sheet_name))
  col_names <- names(sheet)  
  num_sheet_series <- length(col_names)/2
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"), " Number of series to be added: ", num_sheet_series)
  sheet_series <- vector("list", length = num_sheet_series)
  for (i in 1:num_sheet_series) {
    sheet_series <- processSeries(sheet, col_names, sheet_series, i, minimum_length)
  }
  file_series <- c(file_series, sheet_series)
}

ep_file <- "./data/xlsx/uk_centric_multi_asset_class_data_for_simulation_real_returns_010217_daily.xlsx"
sheet_name <- "Sheet1"
message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3")," Processing uk_centric..., sheet: ", sheet_name, "...")
sheet <- rm.emptycol(read_excel(ep_file, sheet=sheet_name, range = cell_rows(c(3,NA))))
col_names <- names(sheet)
num_sheet_series <- length(col_names)/2
message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"), " Number of series to be added: ", num_sheet_series)
sheet_series <- vector("list", length = num_sheet_series)
for (i in 1:num_sheet_series) {
  sheet_series <- processSeries(sheet, col_names, sheet_series, i, minimum_length)
}
file_series <- c(file_series, sheet_series)
file_series[which(sapply(file_series, function(series) series$n) <= minimum_length)] <- NULL # removing series below the minimum length
saveRDS(file_series, file = "./rds/xlsx_file_data_daily.rds")
