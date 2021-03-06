/******/

select distinct industry,Sector, count(distinct Ticker)
  FROM [StockData].[dbo].[CompanyNames] group by Industry,Sector
  order by 2,1

  ;
  with t as (
  select distinct pricedate
				,sum(adjustedclose)/count(distinct ticker) as industry_index 
				,sum(adjustedclose*h.volume)/sum(h.volume) as industry_index2 
				from HistoricalPrices h
  join
  CompanyNames c
  on c.ticker=h.symbol and c.Industry='Industrial Electrical Equipment'
  where Pricedate between '2017-07-31' and GETDATE()
  group by pricedate
  )

  select * from t

  select distinct pricedate
				,Ticker
				,adjustedclose
				from HistoricalPrices h
  join
  CompanyNames c
  on c.ticker=h.symbol and c.Industry='Industrial Electrical Equipment'
  where Pricedate between '2017-07-31' and GETDATE()


  delete  from HistoricalPrices
  where low>-100