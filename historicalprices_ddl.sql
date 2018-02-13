if OBJECT_ID('HistoricalPrices') is not null
drop table stockdata.historicalprices

create table HistoricalPrices(Symbol varchar(100),Pricedate datetime, low float, openprice float, high float, closeprice float,
volume float,[adjustedclose] float,[splitcoefficient] float, dividendamount float)


