/****** Script for SelectTopNRows command from SSMS  ******/


if OBJECT_ID('[StockData].dbo.PriceIndex') is not null
drop table [StockData].dbo.priceindex


create table [StockData].dbo.priceindex (Industry varchar(100),Sector varchar(100),  Pricedate datetime ,priceindex float,volindex float,businessindex float)


insert into StockData.dbo.priceindex

SELECT distinct Industry,Sector,  Pricedate 
, sum(closeprice)/count(distinct Ticker) as priceindex
,sum(h.volume)/count(distinct ticker) as volindex
,(sum(closeprice)/count(distinct Ticker))* sum(h.volume)/count(distinct ticker) as businessindex
  FROM [StockData].[dbo].[CompanyNames] n
  join stockdata.dbo.HistoricalPrices h
  on h.Symbol=n.Ticker
  and datediff(year,h.Pricedate ,GETDATE()) between 0 and 5
  
group by Industry, Sector,Pricedate