from crochet import setup, wait_for
from statsmodels.tsa.ar_model import AR_DEPRECATION_WARN
setup()
import yfinance as yf
import pandas as pd
from scrapy.crawler import CrawlerProcess
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
import time
from textblob import TextBlob 
import spacy
from statsmodels.tsa.stattools import adfuller

import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np

import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import os 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import warnings
import seaborn
warnings.filterwarnings("ignore")

c_name = ""
c_number = ""
n_name = ""

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
             "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
             'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 
             'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
             'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
             'to', 'from', 'up', 'down', 'in', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 
             'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
             'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 's', 't', 'can',
             'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
             'aren', 'couldn', 'didn', 'doesn', 'hadn','hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 
             'shouldn', 'wasn', 'weren','wouldn','may']

stopwords_more = [ "aren't", "haven't", "weren't", "hadn't", "couldn't", "can't", "shan't", "won't", "wasn't", "ain't"
                 ,"mustn't", "wouldn't", "shouldn't", "needn't", "hasn't"]

nlp = spacy.load('en_core_web_sm')
def space(comment):
    doc = nlp(comment)
    return " ".join([token.lemma_ for token in doc])


def get_name():
    global c_name
    print(c_name+"#######################################")
    return c_name

def preprocess_news(newsdata):
    newsdata['Headline'] = newsdata['Headline'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    newsdata['Headline'] = newsdata['Headline'].str.replace('[^\w\s]','')
    newsdata['Headline'] = newsdata['Headline'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    newsdata['Headline'] = newsdata['Headline'].apply(lambda x: " ".join('not' if x in stopwords_more else x for x in x.split()))
    newsdata['Headline'] = newsdata['Headline'].apply(space)
    return newsdata

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess_tweet(tweetdata):
    tweetdata.drop_duplicates(keep="first",inplace=True)
    for index,row in tweetdata.iterrows():
        stre=row["Tweets"]
        my_new_string = re.sub(r"http\S+", "", stre)
        my_new_string = re.sub('[^ a-zA-Z0-9]', '',my_new_string)
        tweetdata.at[index,'Date'] = row["Date"]
        tweetdata.at[index,'Tweets'] = my_new_string
        
    tweetdata['Tweets'] = tweetdata['Tweets'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    tweetdata['Tweets'] = tweetdata['Tweets'].str.replace('[^\w\s]','')
    tweetdata['Tweets'] = tweetdata['Tweets'].apply(lambda x: remove_emoji(x))
    tweetdata['Tweets'] = tweetdata['Tweets'].apply(lambda x: " ".join("not" if x in stopwords_more else x for x in x.split()))
    tweetdata['Tweets'] = tweetdata['Tweets'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    tweetdata['Tweets']= tweetdata['Tweets'].apply(space)  
    return tweetdata                      


@wait_for(timeout=500.0)
def temp(runner,QuotesSpider):
    print("inside temp")
    return runner.crawl(QuotesSpider)


def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

def sentiment_score(df, value):
    sia=SentimentIntensityAnalyzer()
    df['Compound_'+value]=[sia.polarity_scores(v)['compound'] for v in df[value]]
    df['Negative_'+value]=[sia.polarity_scores(v)['neg'] for v in df[value]]
    df['Neutral_'+value]=[sia.polarity_scores(v)['neu'] for v in df[value]]
    df['Positive_'+value]=[sia.polarity_scores(v)['pos'] for v in df[value]]
    return df

def get_sentiment(df):
    global_polarity = 0 
    sentiment = []
    data_length = len(df)
    pos = 0
    neg = 0
    neu = 0
    for text in df:
        blob = TextBlob(text)
        polarity = 0
        for sentence in blob.sentences:
            polarity = sentence.sentiment.polarity
            global_polarity += polarity
    if(data_length != 0):
        global_polarity = global_polarity/data_length
    else:
        global_polarity = global_polarity
    sentiment = np.array(sentiment)
    return pos, neg, neu


def main(name):
    
    #for historical data
    df = pd.read_csv('C:\\Users\\Hp\\Desktop\\IMP\\BEproject\\coding part\\StockMarketPrediction-django\\main\\csv_files\\company.csv')
    print(name)
    global c_name, n_name, c_number 

    #Company name for Twitter data and news data for Business Insider
    c_name = name[0]
    
    #from .historical_data import historical_data
    #historical_data()
    

    # configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
    # runner = CrawlerRunner({
    # 'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    # })
    #if c_name!="":
    #    from .news_data import spiders
    #     temp(runner,spiders.QuotesSpider1)
    #     temp(runner,spiders.QuotesSpider2)
    
    print("*********************************************")

    news1 = pd.read_csv('C:\\Users\\Hp\\Desktop\\IMP\\BEproject\\coding part\\StockMarketPrediction-django\\main\\news1.csv')
    news2 = pd.read_csv('C:\\Users\\Hp\\Desktop\\IMP\\BEproject\\coding part\\StockMarketPrediction-django\\main\\news2.csv')

    #print(news1)
    #merge all news with resp. to dates
    if len(news1)>len(news2):
        news = pd.concat([news1,news2]).reset_index()
    else:
        news = pd.concat([news2,news1]).reset_index()
    

    news.drop_duplicates(keep="first",inplace=True)    
    newsdata = news.groupby('Date')['Headline'].apply(' '.join).reset_index()
    #newsdata

    preprocessed_news = preprocess_news(newsdata)
    
    #from .twitter_data import twitter_data
    #twitter_data()

    tweetdata=pd.read_csv('C:\\Users\\Hp\\Desktop\\IMP\\BEproject\\coding part\\StockMarketPrediction-django\\main\\csv_files\\Tweets.csv')
    
    preprocessed_tweets = preprocess_tweet(tweetdata)
    #recent_tweets = preprocessed_tweets[-7::]

    news_sentiment = sentiment_score(preprocessed_news, "Headline")
    news_sentiment = news_sentiment.iloc[::-1].reset_index(drop = True)
    #print(news_sentiment.head(5))
    
    tweets_sentiment = sentiment_score(preprocessed_tweets,"Tweets")
    #print(tweets_sentiment.tail(5))

    historicaldata = pd.read_csv('C:\\Users\\Hp\\Desktop\\IMP\\BEproject\\coding part\\StockMarketPrediction-django\\main\\csv_files\\historical_data.csv')
    final1 = pd.merge(historicaldata, news_sentiment, how="left", on="Date")
    final = pd.merge(final1,tweets_sentiment,how="left",on="Date")
    data = final
    final.to_csv("C:\\Users\\Hp\\Desktop\\IMP\\BEproject\\coding part\\StockMarketPrediction-django\\main\\csv_files\\stock market analysis data.csv",index=False)

    data = data.fillna(0)
    data['Open-Close'] = (data.Open - data.Close)/data.Open
    data['High-Low'] = (data.High - data.Low)/data.Low
    data['percent_change'] = data['Adj Close'].pct_change()
    data['std_5'] = data['percent_change'].rolling(5).std()
    data['ret_5'] = data['percent_change'].rolling(5).mean()
    data.dropna(inplace=True)

    data1 = data.copy(deep=False)
    prediction_data = data.tail(1)
    data.drop(data.tail(1).index,inplace=True)

    X = data.drop(['Date','Open','High','Low','Close','Headline','Tweets'],axis=1)

    # Y is the target or output variable
    y = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, -1)

    X_Cols = X
    Y_Cols = y

    # Split X and y into X_
    X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, train_size=0.80, test_size=0.20, random_state=100)
  
    rand_frst_clf = RandomForestClassifier(n_jobs= -1, min_samples_leaf= 1, n_estimators= 1000, random_state= 5, criterion= "gini",min_samples_split = 5)
  
    rand_frst_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = rand_frst_clf.predict(X_test)
    
    prediction_data = prediction_data.drop(['Date','Open','High','Low','Close','Headline','Tweets'],axis=1)
    
    ans = rand_frst_clf.predict(prediction_data)[0]
        
    pos_t, neg_t= sum(tweets_sentiment['Positive_Tweets'].tail(7)), sum(tweets_sentiment['Negative_Tweets'].tail(7))
    pos_n, neg_n= sum(news_sentiment['Positive_Headline'].tail(15)), sum(news_sentiment['Negative_Headline'].tail(15))

    neu_t = sum(tweets_sentiment['Neutral_Tweets'].tail(7))
    neu_n = sum(news_sentiment['Neutral_Headline'].tail(15))
    if ans <0:
        neg_t += neu_t
        neg_n += neu_n
    else:
        pos_t += neu_t   
        pos_n += neu_n

    pie1 = ['Positive', pos_t], ['Negative', neg_t]
    pie2 = ['Positive', pos_n], ['Negative', neg_n]

    g1 = [['Date', 'Close']]
    columns = data[['Date', 'Close']]
    g1 = columns.values.tolist()

    dataset = data[['Date', 'Open', 'Low','High', 'Close']].tail(30)
    g2 = dataset.values.tolist()

    data1.Close.dropna(inplace=True)
    #ndiffs(data1.Close, test="adf")

    train_data, test_data = data1[0:int(len(data1)*0.8)], data1[int(len(data1)*0.8):]
    train_ar = train_data['Close'].apply(lambda x: round(x, 3))
    train_ar = train_ar.values

    test_ar = test_data['Close'].apply(lambda x: round(x, 3))
    test_ar = test_ar.values
    history = [x for x in train_ar]

    print(test_data['Close'].tail(1))

    predictions = list()
    for t in range(len(test_ar)):
        model = ARIMA(history, order=(2,1,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_ar[t]
        history.append(obs)
    
    model = ARIMA(history, order=(2,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]

    error = mean_squared_error(test_ar, predictions)
    print('Testing Mean Squared Error: %.3f' % error)
    error2 = smape_kun(test_ar, predictions)
    print('Symmetric mean absolute percentage error: %.3f' % error2)

    print(predictions[-1], test_ar[-1])
    prediction_by_ARIMA = predictions[-1][0]
    #print(len(data1), len(data))
    #print(data1.tail(5))
    #print(data.tail(5))

    print(ans)

    return ans, g1, g2, pie1, pie2, round(yhat[0],2)































#print('Correct Prediction (%): ', accuracy_score(y_test, y_pred, normalize = True) * 100.0)
#print('Correct Prediction (%): ', accuracy_score(y_test, y_pred, normalize = True) * 100.0)