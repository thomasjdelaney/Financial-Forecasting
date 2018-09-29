# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 15:25:25 2018

@author: Thomas Delaney
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import datetime
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

ticker = 'FDRVF'
start_date = datetime.datetime(2013, 7, 1)
end_date = datetime.datetime(2018, 7, 1)
all_days = pd.date_range(start=start_date, end=end_date, freq='B')
f = web.DataReader(ticker, 'morningstar', start_date, end_date)

closing_prices = f['Close'][ticker]

plt.plot(closing_prices, label='FDRVF')
plt.xlabel('Date'); plt.ylabel('Closing Price');
plt.show(block=False)

decomp = seasonal_decompose(closing_prices, model='additive')
decomp.plot()


fit_2 = ExponentialSmoothing(np.asarray(closing_prices)).fit(smoothing_level=0.2)
