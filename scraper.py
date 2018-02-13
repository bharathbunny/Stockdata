from bs4 import BeautifulSoup
import requests
import re
import urllib3


def get_table_headers():
    
    url = "https://www.finviz.com/screener.ashx?v=111"
    
    response = requests.get( url)
    
    soup = BeautifulSoup(response.content) 
    table_headers = []
    for th in soup.select(".table-top"):
        table_headers.append(th.get_text())
    table_headers.insert(1, "Ticker")
    return table_headers


def get_rows_from_soup(soup, table_headers):
	table_row_data = []
	counter = 0
	row_data = {}
	for tr in soup.select(".screener-body-table-nw"):
		row_data[table_headers[counter]] = tr.get_text()
		counter += 1
		if counter >= len(table_headers):
			counter = 0
			table_row_data.append(row_data)
			row_data = {}
	return table_row_data


def get_data():
    headers = get_table_headers()
    all_data = []
    ended = False
    intitial_number=1
    while not ended:
        url="https://www.finviz.com/screener.ashx?v=111&r=%d"%(intitial_number)
        
        response=requests.get(url)
        soup=BeautifulSoup(response.content)
        all_data +=get_rows_from_soup(soup,headers)
        print(len(all_data))
        intitial_number +=20
        if not re.findall(b"<b>next</b>", response.content):
            ended=True
    return all_data
	
    
    
    
#    while not ended:
#        url = "http://www.finviz.com/screener.ashx?v=111&f=cap_nano&r={}".format(initial_number)
#        content = urllib.request.urlopen(url).read()
#		soup = BeautifulSoup(content)
#		all_data += get_rows_from_soup(soup, headers)
#		print(len(all_data))
#		initial_number += 20
#		print(type(content))
#		print(initial_number)
#		if not re.findall(b"<b>next</b>", content):
#			ended = True
#	return all_data
#

# need to determine which table I am trying to access
# need to get headers from that table<td>, and return them in a list as text
# then need to get td of stocks with a market cap under 50M
# need to scrape all the same tables from all pages
# then need to load that data into a postgres db table
# then need to query the db 
# then need to get the avg