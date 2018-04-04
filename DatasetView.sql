
/****** Script for SelectTopNRows command from SSMS  ******/
CREATE VIEW dbo.vwdataset AS
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
 --and p.Symbol= 'fb'
  and p.Pricedate<=h.Pricedate
  and datediff(day,h.Pricedate,getdate()) between 0 and 365*4
  )t
  where rn <=10


  )
  select ticker,pricedate, busin_1,busin_2,busin_3,busin_4, busin_5, price_1, price_2,price_3,
         price_4,   price_5,priceindex_1, priceindex_2, priceindex_3, priceindex_4, priceindex_5,
         vol1,vol2,    vol3,   vol4, vol5,out_1, out_3, out_5, out_7, out_10, out_15, out_30 from (
  select distinct t.Ticker,t.Pricedate,p.businessindex, 'busin_'+convert(varchar,t.rn) as varname from
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
   select distinct t.Ticker,t.Pricedate,p.priceindex, 'priceindex_'+convert(varchar,t.rn) as varname from t
  join

 [StockData].[dbo].[CompanyNames] c
  on c.Ticker=t.Ticker
  join
  stockdata.dbo.priceindex p
  on p.Industry=c.Industry
  and p.Sector=c.Sector
  and t.prev_price=p.Pricedate
   union
   select distinct t.Ticker,t.Pricedate,p.volindex, 'vol'+convert(varchar,t.rn) as varname from t
  join

 [StockData].[dbo].[CompanyNames] c
  on c.Ticker=t.Ticker
  join
  stockdata.dbo.priceindex p
  on p.Industry=c.Industry
  and p.Sector=c.Sector
  and t.prev_price=p.Pricedate

  union 
  select t.Ticker,t.Pricedate,t.avgprice,t.varname from t

  union

  select Ticker,Pricedate,avgprice,varname from 
  
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
)a
pivot
(max(businessindex) for  varname in ( busin_1,busin_2,busin_3,busin_4, busin_5, price_1, price_2,price_3,
         price_4,   price_5,priceindex_1, priceindex_2, priceindex_3, priceindex_4, priceindex_5,
         vol1,vol2,    vol3,   vol4, vol5,out_1, out_3, out_5, out_7, out_10, out_15, out_30))pvt