from .result import c_number, c_name, n_name
import pandas as pd
import scrapy    
from scrapy.crawler import CrawlerProcess
import logging 
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
import pathlib
import datetime
import dateparser
import time
from textblob import TextBlob 
import spacy
class spiders():
    class QuotesSpider1(scrapy.Spider):  
        name = "news"
        start_urls = []
        start_urls.append("https://www.businessinsider.in/searchresult.cms?query={}&sortorder=score".format(c_name))
        for start in range(0,25):
            url = 'https://www.businessinsider.in/searchresult.cms?query={}&sortorder=score&curpg={}'.format(c_name,start)
            start_urls.append(url)
        #print(start_urls)
        custom_settings = {
            
            'LOG_LEVEL': logging.WARNING,
            'CONCURRENT_REQUESTS':96,
            "FEEDS": {
                #"news.json": {"format": "json", "overwrite": True},
                pathlib.Path('news1.csv'): {
                    'format': 'csv',
                    'fields': ['Date','Headline'],
                    'overwrite': True,
                },
            },
        }
        

        def parse(self, response):

            for new in response.css('.list-bottom-story'):
                temp = new.css('div.list-timestamp::text').extract()
                yield {
                    'Date':dateparser.parse(str(temp)).strftime("%Y-%m-%d"),
                    'Headline': new.css('h2.list-bottom-small-title a::text').extract(),  
                }


    class QuotesSpider2(scrapy.Spider):  
        name = "news"
        start_urls = ['https://www.business-standard.com/advance-search?advance=Y&type=news&c-cname=cname&cname={}&company={}&itemsPerPage=19&page={}'.format(n_name,c_number, 1)]
        #print(*start_urls,sep="\n")
       
        custom_settings = {
            
            'LOG_LEVEL': logging.WARNING,
            'CONCURRENT_REQUESTS':96,
            "FEEDS": {
                #"news.json": {"format": "json", "overwrite": True},
                pathlib.Path('news2.csv'): {
                    'format': 'csv',
                    'fields': ['Date','Headline'],
                    'overwrite': True,
                },
            },
        }
        
        def parse(self, response):
            for new in response.css('.listing li'):
                lists = new.css('p::text').extract()
                #print(lists)
                yield {
                    'Date':dateparser.parse(lists[0]).strftime("%Y-%m-%d"),
                    'Headline':lists[1]
                }
                            
            NEXT_PAGE_SELECTOR = '.colum-nextPrev div.next-colum a::attr(href)' 
            next_page = response.css(NEXT_PAGE_SELECTOR).extract()[0]
            url = 'https://www.business-standard.com'
            #print(url+next_page)
            if (next_page):
                    yield scrapy.Request(response.urljoin(url+next_page), callback=self.parse)
        
