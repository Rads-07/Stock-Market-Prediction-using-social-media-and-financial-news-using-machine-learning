from .result import c_number, c_name, n_name
import datetime
import pandas as pd
import snscrape.modules.twitter as sntwitter

def twitter_data():
    today = datetime.date.today()
    until = today.strftime("%Y-%m-%d")
    since = (today - datetime.timedelta(days=4*365)).strftime("%Y-%m-%d")

    tweets_list = []
    # # Using TwitterSearchScraper to scrape data and append tweets to list
    for tweet in sntwitter.TwitterSearchScraper('#{} stocks until:{}'.format(c_name,until)).get_items():
        tweet.date = datetime.datetime.strptime(str(tweet.date).split(" ")[0], "%Y-%m-%d").strftime("%Y-%m-%d")
        #print(tweet.date)
        if tweet.date < since:
            break
        tweets_list.append([tweet.date, tweet.content])
    
    df = pd.DataFrame(tweets_list, columns=['Date', 'Tweets'])
    df = df.groupby('Date')['Tweets'].apply(' '.join).reset_index()
    df.to_csv('C:\\Users\\Hp\\Desktop\\IMP\\BEproject\\coding part\\StockMarketPrediction-django\\main\\csv_files\\Tweets.csv')