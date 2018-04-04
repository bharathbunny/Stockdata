# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 22:38:26 2018

@author: Yamini
"""


import pandas as pd
import alpha_vantage as av
from alpha_vantage.timeseries import TimeSeries
import bt


import pypyodbc


#%%

cnxn = pypyodbc.connect("DRIVER={SQL Server};SERVER=DESKTOP-38U09HS\SQLEXPRESS;DATABASE=StockData")

cursor=cnxn.cursor()

#%%

sql="""

select * from historicalprices where symbol='MSFT' and pricedate between '2015-01-01' and '2016-08-01'

"""

msft=pd.read_sql(sql,cnxn)
msft.index=msft.pricedate
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
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
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=50, random_state=rng)


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
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=50, random_state=rng)


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


#%%
sql="""


/****** Script for SelectTopNRows command from SSMS  ******/
;
with t as(
select * from (
SELECT distinct  c.Ticker
				,h.Pricedate
				, p.Pricedate as prev_price
				,(p.openprice+p.closeprice+p.adjustedclose+p.high+p.low)/5 as avgprice
				,  ROW_NUMBER() over (partition by c.ticker,h.pricedate order by p.pricedate desc) as rn
				,'price_'+convert(varchar,ROW_NUMBER() over (partition by c.ticker,h.pricedate order by p.pricedate desc) ) as varname
				 
  FROM [StockData].[dbo].[CompanyNames] c
  join
  StockData.dbo.HistoricalPrices h
  on h.Symbol=c.Ticker
  join
  stockdata.dbo.HistoricalPrices p
  on p.Symbol=h.Symbol
 and p.Symbol= 'cmc'
  and p.Pricedate<=h.Pricedate
  and datediff(day,h.Pricedate,getdate()) between 1 and 365*4
  )t
  where rn <=10


  )

  select distinct t.Ticker,t.Pricedate,t.prev_price,p.businessindex,t.rn, 'busin_'+convert(varchar,t.rn) as varname from
   t
  join

 [StockData].[dbo].[CompanyNames] c
  on c.Ticker=t.Ticker
  join
  stockdata.dbo.priceindex2 p
  on p.Industry=c.Industry
  and p.Sector=c.Sector
  and t.prev_price=p.Pricedate

  union
   select distinct t.Ticker,t.Pricedate,t.prev_price,p.priceindex,t.rn, 'priceindex_'+convert(varchar,t.rn) as varname from t
  join

 [StockData].[dbo].[CompanyNames] c
  on c.Ticker=t.Ticker
  join
  stockdata.dbo.priceindex p
  on p.Industry=c.Industry
  and p.Sector=c.Sector
  and t.prev_price=p.Pricedate
   union
   select distinct t.Ticker,t.Pricedate,t.prev_price,p.volindex,t.rn, 'vol'+convert(varchar,t.rn) as varname from t
  join

 [StockData].[dbo].[CompanyNames] c
  on c.Ticker=t.Ticker
  join
  stockdata.dbo.priceindex p
  on p.Industry=c.Industry
  and p.Sector=c.Sector
  and t.prev_price=p.Pricedate

  union 
  select * from t

  union

  select * from 
  
  (select distinct a.Ticker
					,a.Pricedate
					,h.Pricedate as next_price
					, (h.openprice+h.closeprice+h.adjustedclose+h.high+h.low)/5 as avgprice
					,  ROW_NUMBER() over (partition by a.ticker,a.pricedate order by h.pricedate ) as rn
				,'out_'+convert(varchar,ROW_NUMBER() over (partition by a.ticker,a.pricedate order by h.pricedate ) ) as varname
				 
					from (select distinct t.pricedate,t.Ticker from t) a
  join
  StockData.dbo.HistoricalPrices h
  on a.Ticker=h.Symbol
  and a.Pricedate<h.Pricedate) out_val
  where rn in (1,3,5,7,10,15,30)

  order by 1,2,varname
"""

stckvals=pd.read_sql(sql,cnxn)

#%%
temp2=pd.pivot_table(stckvals,columns=['varname'],values=['businessindex'],index=['ticker','pricedate'],aggfunc=np.max)
temp2.reset_index(level=0, inplace=True)
temp2.reset_index(level=0, inplace=True)
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier
# Create the dataset
rng = np.random.RandomState(1)
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=10, random_state=rng)


temp2.columns=[t[1] if 'businessindex' in t else t[0] for t in temp2.columns.tolist()]
#for al in ['busin_1', 'busin_2', 'busin_3']:
#    temp2[al]=(temp2[al]-np.mean(temp2[al].values))/(np.std(temp2[al]))
temp2[ ['busin_1', 'busin_2', 'busin_3']]=np.log(temp2[ ['busin_1', 'busin_2', 'busin_3']])
outs=['out_1', 'out_10', 'out_15', 'out_3', 'out_30', 'out_5', 'out_7']
tgt='out_5'
c=temp2.columns.tolist()
[c.remove(x) for x in outs]

c.remove('pricedate')
c.remove('ticker')
c=[
   'busin_1',
 'busin_2',
 'busin_3',
# 'busin_4',
# 'busin_5',
 'price_1',
 'price_2',
 'price_3',
 'price_4',
 'price_5',
# 'priceindex_1',
# 'priceindex_2',
# 'priceindex_3',
# 'priceindex_4',
# 'priceindex_5',
 'vol1',
 'vol2',
 'vol3',
 'vol4',
# 'vol5'
]
#
#c=['busin_1',  'busin_2',
#       'busin_3', 'busin_4', 'busin_5', 'busin_6', 'busin_7', 'busin_8',
#       'busin_9',  'price_1', 'price_10', 'price_2', 'price_3',
#       'price_4', 'price_5', 'price_6', 'price_7', 'price_8', 'price_9',
#       'priceindex_1', 'priceindex_10', 'priceindex_2', 'priceindex_3',
#       'priceindex_4', 'priceindex_5', 'priceindex_6', 'priceindex_7',
#       'priceindex_8', 'priceindex_9', 'vol1', 'vol10', 'vol2', 'vol3',
#       'vol4', 'vol5', 'vol6', 'vol7', 'vol8', 'vol9']

temp3=temp2[temp2[tgt].notnull()].copy()

regr_2.fit(temp3.loc[:300,c], temp3.loc[:300,tgt])

outvals=regr_2.predict(temp3.loc[300:,c])
scr=regr_2.score(temp3.loc[300:,c], temp3.loc[300:,tgt])
plt.plot(temp3.pricedate,temp3[tgt],label='training')
plt.plot(temp3.loc[300:,:].pricedate,temp3.loc[300:,tgt],label='true vals')
plt.plot(temp3.loc[300:,:].pricedate,outvals,label='predicted')

plt.grid()



alpha = 0.95

clf = GradientBoostingRegressor(loss='ls', alpha=alpha,
                                n_estimators=6, max_depth=2,
                                learning_rate=.2, min_samples_leaf=9,
                                min_samples_split=9)
clf.fit(temp3.loc[:300,c], temp3.loc[:300,tgt])
# Make the prediction on the meshed x-axis
#y_upper = clf.predict(temp3.loc[300:,c])
#
#clf.set_params(alpha=1.0 - alpha)
#clf.fit(temp3.loc[:300,c], temp3.loc[:300,tgt])
#
## Make the prediction on the meshed x-axis
#y_lower = clf.predict(temp3.loc[300:,c])
#
#clf.set_params(loss='ls')
#clf.fit(temp3.loc[:300,c], temp3.loc[:300,tgt])

# Make the prediction on the meshed x-axis
y_pred = clf.predict(temp3.loc[300:,c])

#plt.fill_between(np.arange(189),y_lower,y_upper,alpha=0.5,label='CI')
#plt.plot(temp3.loc[300:,:].pricedate,y_upper,label='upper')
plt.plot(temp3.loc[300:,:].pricedate,y_pred,label='predicted_gradient')
#plt.plot(temp3.loc[300:,:].pricedate,y_lower,label='lower')
plt.legend()
print(clf.score(temp3.loc[300:,c],temp3.loc[300:,tgt]))
print(scr)


#%%

from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.svm import SVC
from sklearn import linear_model
tgt='out_clas'
tgt2='out_5'
temp3['out_clas']=[1 if t>0 else 0 for t in temp3.price_1-temp3[tgt2]]



alpha = 0.95

clf = GradientBoostingClassifier(loss='deviance', verbose=1,
                                n_estimators=6, max_depth=2,
                                learning_rate=.2, min_samples_leaf=9,
                                min_samples_split=9)
#clf = SVC(probability=True)
#clf = linear_model.SGDClassifier(loss='modified_huber')
clf.fit(temp3.loc[:300,c], temp3.loc[:300,tgt])
# Make the prediction on the meshed x-axis
#y_upper = clf.predict(temp3.loc[300:,c])

# Make the prediction on the meshed x-axis
y_pred = clf.predict_proba(temp3.loc[300:,c])[:,1]

roc_auc_score(temp3.loc[300:,tgt], y_pred)






