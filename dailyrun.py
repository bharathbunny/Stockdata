# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:03:08 2018

@author: Yamini
"""



import pandas as pd
import alpha_vantage as av
from alpha_vantage.timeseries import TimeSeries
#import bt


import pypyodbc
#%%

cnxn = pypyodbc.connect("DRIVER={SQL Server};SERVER=DESKTOP-38U09HS\SQLEXPRESS;DATABASE=StockData")

cursor=cnxn.cursor()
#%%

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc,roc_curve
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



def model_classification(temp3,c,outs):
    temp3.sort_values('pricedate',inplace=True)
    train_length=int(len(temp3)*0.65)
    logreg=LogisticRegression()
    possible=False
    model=[]
    th_classify=0.1
    while ((~possible) & (th_classify>=0.05)):
        print(th_classify)
        temp3['case_control']=[int(p) for p in list((temp3[outs].fillna(0).max(axis=1)-temp3.price_1)/temp3.price_1 >=th_classify)]
        if(temp3.case_control.sum()/len(temp3)>=0.08):
            possible=True
        else:
            th_classify=th_classify-0.01
    if(th_classify<0.05):
        return(False,model,th_classify,0)
    else:
        logreg.fit(temp3.loc[:train_length,c],temp3.loc[:train_length,'case_control'])
        probs=logreg.predict_proba(temp3.loc[train_length:,c])
        fp,tp,th=roc_curve(temp3.loc[train_length:,'case_control'],probs[:,1])
        
        return(True,logreg,th_classify,auc(fp,tp))


def pick_best(data,outs,c,top_n=0,classify=False,classify_var='None',classify_outs=[]):
    model=[]
    rslts=[]
    outdata=data.loc[data.pricedate.values.argmax(),:].copy()
#    print(outdata)
    if(classify):
        temp3=data[data[classify_var].notnull()].copy()
       
        return(model_classification(temp3,c,classify_outs))
    else:
            
        for tgt in outs:
            temp3=data[data[tgt].notnull()].copy()
            scr1,scr2,regr_2,clf=models(temp3,c,tgt)
    #        print(tgt,scr1,scr2)
            if scr1>scr2:
                model=regr_2
            else:
                model=clf
           
            rslts.append([outdata['ticker'],tgt, model.predict(np.reshape(outdata[c].values.T,newshape=(1,-1)))[0],outdata.loc['price_1'],max(scr1,scr2)])
        if(top_n==0):
            return(rslts[np.argmax([(i[2]-i[3])* np.max([0,i[4]]) for i in rslts])])
        else:
            return([rslts[i] for i in np.argsort([-1*(i[2]-i[3])* np.max([0,i[4]]) for i in rslts])[:np.min([top_n,len(rslts)])]])
#%%

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

outs=[
      'out_1',
      'out_10', 'out_15', 'out_3', 'out_30', 'out_5', 'out_7']

data=pd.read_sql("""select * FROM [StockData].[dbo].[vwdataset]  where ticker in ('cmc') and pricedate>'2016-01-01' """,cnxn)

g=pick_best(data,outs,c)
    

#%%
sql="""

  select distinct d.* FROM [StockData].[dbo].[vwdataset] d
  where d.ticker in ('ATTU'
,'EVOL'
,'ICAD'
)
"""

data_all=pd.read_sql(sql,cnxn)
#%%
g=[]

for i in ['ATTU','EVOL','ICAD']:
    data=data_all[data_all.ticker==i].copy()
    
    data.reset_index(inplace=True)
    outdata=data.loc[data.pricedate.values.argmax(),:].copy()
#    g.extend(pick_best(data,outs,c,3))
    possible,model,thresh,scr=pick_best(data,outs,c,3,True, 'out_10',[  'out_1','out_10', 'out_15', 'out_3', 'out_30', 'out_5', 'out_7'])
    prb=model.predict_proba(np.reshape(outdata[c].values.T,newshape=(1,-1)))[:,1]
    if(thresh>=0.05):
        print(i,', AUC: ',scr,', Profit:',thresh,' Price:',outdata.loc['price_1'])
        print('Prob: ',prb,', Confidence:',prb*scr)
    else:
        print(i, ' Not possible for classification')
