from .result import c_number, c_name, n_name
import yfinance as yf
import datetime
import pandas as pd

def historical_data():
    df = pd.read_csv("C:\\Users\\Hp\\Desktop\\IMP\\BEproject\\coding part\\StockMarketPrediction-django\\main\\csv_files\\company.csv")
    a = df[df["company_name"] == c_name]

    ticker = a.ticker_name.item()
    today = datetime.date.today()
    end = today.strftime("%Y-%m-%d")
    start = (today - datetime.timedelta(days=4*365)).strftime("%Y-%m-%d")
    print("hello")
    dataset = yf.download(ticker, start=start, end = end)

    dataset.dropna(axis=0, how = "all")
    dataset.fillna(dataset.mean(axis=0))
    dataset.reset_index(inplace=True)     
    dataset.to_csv("C:\\Users\\Hp\\Desktop\\IMP\\BEproject\\coding part\\StockMarketPrediction-django\\main\\csv_files\\historical_data.csv",index=False)
