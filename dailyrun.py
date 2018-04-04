# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:03:08 2018

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
 and p.Symbol= %s
  and p.Pricedate<=h.Pricedate
  and datediff(day,h.Pricedate,getdate()) between 1 and 365*2
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

sql2="""


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
 and p.Symbol= %s
  and p.Pricedate<=h.Pricedate
  and datediff(day,h.Pricedate,getdate()) between 1 and 365*2
  join (select max(pricedate) pdt from StockData.dbo.HistoricalPrices where Symbol=%s) f
  on f.pdt=h.Pricedate

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
select distinct t.Ticker,t.Pricedate,t.pricedate
,(h.openprice+h.closeprice+h.adjustedclose+h.high+h.low)/5 as avgprice,1, 'pred' 
as varname from t
join
StockData.dbo.HistoricalPrices h
  on h.Symbol=t.Ticker
  and h.pricedate=t.pricedate

  
  order by 1,2,varname

"""

#%%
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
rng = np.random.RandomState(1)


def models(temp3,c,tgt):
    temp3.sort_values('pricedate',inplace=True)
    train_length=int(len(temp3)*0.65)
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=10, random_state=rng)

    regr_2.fit(temp3.loc[:train_length,c], temp3.loc[:train_length,tgt])

#    outvals=regr_2.predict(temp3.loc[300:,c])
    scr1=regr_2.score(temp3.loc[train_length:,c], temp3.loc[train_length:,tgt])
    
    
    
    
    alpha = 0.95
    clf = GradientBoostingRegressor(loss='ls', alpha=alpha,
                                    n_estimators=10, max_depth=3,
                                    learning_rate=.2, min_samples_leaf=9,
                                    min_samples_split=9)
    clf.fit(temp3.loc[:train_length,c], temp3.loc[:train_length,tgt])
    # Make the prediction on the meshed x-axis
#    y_upper = clf.predict(temp3.loc[300:,c])
    
    

    
    scr2=clf.score(temp3.loc[train_length:,c],temp3.loc[train_length:,tgt])
    return scr1,scr2,regr_2,clf






def pick_best(data,outs,c):
    model=[]
    rslts=[]
    outdata=data.loc[data.pricedate.values.argmax(),:].copy()
#    print(outdata)
    for tgt in outs:
        temp3=data[data[tgt].notnull()].copy()
        scr1,scr2,regr_2,clf=models(temp3,c,tgt)
#        print(tgt,scr1,scr2)
        if scr1>scr2:
            model=regr_2
        else:
            model=clf
       
        rslts.append([outdata['ticker'],tgt, model.predict(np.reshape(outdata[c].values.T,newshape=(1,-1)))[0],outdata.loc['price_1'],max(scr1,scr2)])
    return(rslts[np.argmax([(i[2]-i[3])*np.max([0,i[4]]) for i in rslts])])


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

outs=['out_1', 'out_10', 'out_15', 'out_3', 'out_30', 'out_5', 'out_7']

data=pd.read_sql("""select * FROM [StockData].[dbo].[vwdataset]  where ticker in ('P')""",cnxn)

g=pick_best(data,outs,c)
    



