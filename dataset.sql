/****** Script for SelectTopNRows command from SSMS  ******/

if OBJECT_ID('[StockData].dbo.dataset') is not null
drop table [StockData].dbo.dataset


create table [StockData].dbo.dataset (Ticker varchar(100),Pricedate datetime,prev_price datetime, vals float, varname varchar(100))

if OBJECT_ID('tempdb..#temp') is not null
drop table #temp

select * into #temp from (
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
 --and p.Symbol= 'AAME'
  and p.Pricedate<h.Pricedate
  and datediff(day,h.Pricedate,getdate()) between 1 and 600
  )t
  where rn <=5



insert into StockData.dbo.dataset



  select distinct t.Ticker,t.Pricedate,t.prev_price,p.businessindex as vals, 'busin_'+convert(varchar,t.rn) as varname from
 #temp  t
  join

 [StockData].[dbo].[CompanyNames] c
  on c.Ticker=t.Ticker
  join
  stockdata.dbo.priceindex p
  on p.Industry=c.Industry
  and p.Sector=c.Sector
  and t.prev_price=p.Pricedate

  union
   select distinct t.Ticker,t.Pricedate,t.prev_price,p.priceindex, 'priceindex_'+convert(varchar,t.rn) as varname from  #temp t
  join

 [StockData].[dbo].[CompanyNames] c
  on c.Ticker=t.Ticker
  join
  stockdata.dbo.priceindex p
  on p.Industry=c.Industry
  and p.Sector=c.Sector
  and t.prev_price=p.Pricedate
   union
   select distinct t.Ticker,t.Pricedate,t.prev_price,p.volindex, 'vol'+convert(varchar,t.rn) as varname from  #temp t
  join

 [StockData].[dbo].[CompanyNames] c
  on c.Ticker=t.Ticker
  join
  stockdata.dbo.priceindex p
  on p.Industry=c.Industry
  and p.Sector=c.Sector
  and t.prev_price=p.Pricedate

  union 
  select * from  #temp t

  union

  select out_val.Ticker,out_val.Pricedate,out_val.next_price,out_val.avgprice,out_val.varname from 
  
  (select distinct a.Ticker
					,a.Pricedate
					,h.Pricedate as next_price
					, (h.openprice+h.closeprice+h.adjustedclose+h.high+h.low)/5 as avgprice
					,  ROW_NUMBER() over (partition by a.ticker,a.pricedate order by h.pricedate ) as rn
				,'out_'+convert(varchar,ROW_NUMBER() over (partition by a.ticker,a.pricedate order by h.pricedate ) ) as varname
				 
					from (select distinct t.pricedate,t.Ticker from  #temp t) a
  join
  StockData.dbo.HistoricalPrices h
  on a.Ticker=h.Symbol
  and a.Pricedate<h.Pricedate) out_val
  where rn in (1,3,5,7,10,15,30)

  order by 1,2,varname
