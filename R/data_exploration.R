fin_ns <- sapply(fin_series, function(time_series) time_series$n)

isPeriod <- function(time_series, period_str) {time_series$period == period_str}

daily_fin_series_ns = fin_ns[which(sapply(fin_series, function(time_series)isPeriod(time_series,"Daily")))]
weekly_fin_series_ns = fin_ns[which(sapply(fin_series, function(time_series)isPeriod(time_series,"Weekly")))]
monthly_fin_series_ns = fin_ns[which(sapply(fin_series, function(time_series)isPeriod(time_series,"Monthly")))]
quarterly_fin_series_ns = fin_ns[which(sapply(fin_series, function(time_series)isPeriod(time_series,"Quarterly")))]
yearly_fin_series_ns = fin_ns[which(sapply(fin_series, function(time_series)isPeriod(time_series,"Yearly")))]

par(mfrow=c(5,1)) 
daily_hist <- hist(daily_fin_series_ns, xlim=c(0,5000), col=rgb(1,0,0), xlab="Series Length", ylab="No.of Series", main="Number of Daily time series = 1559")
weekly_hist <- hist(weekly_fin_series_ns, xlim=c(0,5000), col=rgb(0,1,0), xlab="Series Length", ylab="No.of Series", main="Number of Weekly time series = 164")
monthly_hist <- hist(monthly_fin_series_ns, xlim=c(0,5000), col=rgb(0,0,1), xlab="Series Length", ylab="No.of Series", main="Number of Monthly time series = 10987")
quarterly_hist <- hist(quarterly_fin_series_ns, xlim=c(0,5000), col=rgb(1,1,0), xlab="Series Length", ylab="No.of Series", main="Number of Quarterly time series = 5305")
yearly_hist <- hist(yearly_fin_series_ns, xlim=c(0,5000), col=rgb(1,0,1), ylab="No.of Series", main="Number of Yearly time series = 6519")
