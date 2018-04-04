# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 00:37:49 2017

@author: Yamini
"""

import pandas as pd
import alpha_vantage as av
from alpha_vantage.timeseries import TimeSeries
import bt
#%%
stocks=pd.read_csv('symbols.txt',sep='|')
ts=TimeSeries(key='AIGN7DB0PZGS3C4X', output_format='pandas')

#%%
data2,meta=ts.get_daily_adjusted(symbol='ACUR',outputsize='full')

data=bt.get('BTC',start='2017-10-01')

#%%
import scraper

data=get_data()


#%%
data=pd.DataFrame(data)
#%%

import pypyodbc

cnxn = pypyodbc.connect("DRIVER={SQL Server};SERVER=DESKTOP-38U09HS\SQLEXPRESS;DATABASE=StockData")

cursor=cnxn.cursor()



#%%
data=pd.read_sql("  select distinct ticker from companynames except select distinct symbol from historicalprices",cnxn)
for i in data.ticker:
    try:
        data2,meta=ts.get_daily_adjusted(symbol=i,outputsize='full')
        data2['symbol']=str(i)
        data2.reset_index(level=0,inplace=True)
        cols=['symbol','Date', 'low', 'open', 'high', 'close', 'volume', 'adjusted close', 'split coefficient', 'dividend amount' ]
        data2=data2[cols]
        cursor.executemany(""" insert into historicalprices values(?,?,?,?,?,?,?,?,?,?) """,[list(c) for c in data2[cols].values])
        cnxn.commit()
    except:
        continue

#%%update daily
import numpy as np    
data=pd.read_sql("  select distinct symbol,max(pricedate) pricedate from historicalprices  group by  symbol",cnxn)
for i in data.symbol:
    try:
        data2,meta=ts.get_daily_adjusted(symbol=i,outputsize=10)
        data2['symbol']=str(i)
        data2.reset_index(level=0,inplace=True)
        cols=['symbol','Date', 'low', 'open', 'high', 'close', 'volume', 'adjusted close', 'split coefficient', 'dividend amount' ]
        data2=data2[cols]
        data2=data2[[np.datetime64(j) for j in data2.loc[data2.symbol==i,'Date'].values]>data.loc[data.symbol==i,'pricedate'].values[0]]
        cursor.executemany(""" insert into historicalprices values(?,?,?,?,?,?,?,?,?,?) """,[list(c) for c in data2[cols].values])
        cnxn.commit()
    except:
        continue
 #%%
x=        ['Ticker', 'Company', 'Country', 'Industry', 'Market Cap',  'Sector',  'Volume']

data=data[x]

data.to_csv('companynames.csv',index=False)
#%%
sql="""

select * from historicalprices where symbol='aapl' and pricedate between '2015-01-01' and '2016-08-01'

"""

msft=pd.read_sql(sql,cnxn)
msft.index=msft.pricedate
#%%
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#msft.plot(x='pricedate',y=['closeprice','low','high'])
autocorrelation_plot(msft.closeprice)
pyplot.show()

#%%
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# fit model
model = ARIMA(msft.closeprice, order=(2,1,0))
model_fit = model.fit()
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

#%%
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
msft['loghigh']=np.log(msft.high)
X =msft.high
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
ci=[]
for t in range(len(test)):
    model = ARIMA(history, order=(3,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    ci.append(output[2])
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
#    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
#test['predicted']=[x[0] for x in predictions]
pyplot.plot((test.values),color='blue')
pyplot.plot((predictions), color='red')
ci=np.reshape(ci,newshape=(136,2))
#pyplot.fill_between(range(len(test)),ci[:,0],ci[:,1],color='grey')
pyplot.show()

#%%

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
# Create the dataset
rng = np.random.RandomState(1)
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=500, random_state=rng)


history=[x for x in train]

x1=history[:-5]
x2=history[1:-4]
x3=history[2:-3]
x4=history[3:-2]
x5=history[4:-1]

history2=history[5:]

X=np.vstack((x1,x2,x3,x4,x5))
X=X.T


regr_1.fit(np.log(X),history2)

regr_2.fit(np.log(X),history2)

plt.figure()
#plt.scatter(np.arange(len(history2)), history2, c="k", label="training samples")

test=msft.high[size-5:]

x1=test[:-5]
x2=test[1:-4]
x3=test[2:-3]
x4=test[3:-2]
x5=test[4:-1]

history2=test[5:]

X=np.vstack((x1,x2,x3,x4,x5))
X=X.T


y_1=regr_1.predict(np.log(X))

y_2=regr_2.predict(np.log(X))


plt.plot(np.arange(len(history2)), history2, c="b", label="testing samples")

#plt.plot(np.arange(len(y_1)), y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(np.arange(len(y_1)), y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
#%%
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
# Create the dataset
rng = np.random.RandomState(1)
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=500, random_state=rng)


history=[x for x in train]

x1=history[:-9]
x2=history[1:-8]
x3=history[2:-7]
x4=history[3:-6]
x5=history[4:-5]
x6=history[5:-4]
history2=history[9:]

X=np.vstack((x1,x2,x3,x4,x5))
X=X.T


regr_1.fit(np.log(X),history2)

regr_2.fit(np.log(X),history2)

plt.figure()
#plt.scatter(np.arange(len(history2)), history2, c="k", label="training samples")

test=msft.high[size-5:]
x1=test[:-9]
x2=test[1:-8]
x3=test[2:-7]
x4=test[3:-6]
x5=test[4:-5]
x6=test[5:-4]
history2=test[5:-4]

X=np.vstack((x1,x2,x3,x4,x5))
X=X.T


y_1=regr_1.predict(np.log(X))

y_2=regr_2.predict(np.log(X))


plt.plot(np.arange(len(history2)), history2, c="b", label="testing samples")

#plt.plot(np.arange(len(y_1)), y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(np.arange(len(y_1)), y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()




