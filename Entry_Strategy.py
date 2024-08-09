import yfinance as yf
import numpy as numpy
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
import pandas_ta as ta
import numpy as np
import scipy.stats as st
from Send_Email import SendText


class Entry_Strategy:

    def __init__(self):

        df_nasdaq = pd.read_csv("Nasdaq_Tickers.csv")
        df_nyse = pd.read_csv("NYSE_Tickers.csv")

        df = pd.concat([df_nasdaq, df_nyse])

        df = df.drop_duplicates(subset="Symbol", keep="first")

        self.end_time = dt.datetime.now()
        #self.end_time = dt.datetime(2024, 1, 23) #for back testing 
        self.start_time = self.end_time - relativedelta(months=3)

        self.tickers=list(df["Symbol"])

        #self.buy_tickers={"Symbol":[], "TI":[], "metric":[], "stats":[]}
        self.buy_tickers={"Symbol":[], "TI":[], "metric":[]}
        self.errors={"Symbol":[], "Error":[], "Date":[]}

        self.current_tickers=pd.read_csv("current_positions.csv")

        self.current_market = yf.download("^GSPC", start=self.start_time, end=self.end_time, progress=False)["Adj Close"].pct_change().dropna()[-1]
        self.market_std = []
        self.market_percent_flips = []
        self.market_skew = []
        self.market_kurtosis = []


    #define functions to be used on market aggragate as well as indiviual stocks
    def compute_std(self, pct_change):
        return np.std(pct_change)

    def compute_percent_flips(self, pct_change):

        return pct_change.rolling(2).apply(lambda x: np.sign(x[0]) != np.sign(x[1])).value_counts()[1]/(len(pct_change)-1)
    
    def compute_skew(self, pct_change):

        return st.skew(pct_change)

    def compute_kurt(self, pct_change):

        return st.kurtosis(pct_change)

    def stats(self):
        
        for symbol in self.buy_tickers["Symbol"]:

            #initialize a blank dict to store statstics 
            stats = {"current price":None, "high":None, "low":None, "standard deviation":None, "beta":None, "percent of flips":None, "skew": None, "percent change mean":None,"previous percent change":None,
                      "sharpe ratio":None, "kurtosis":None, "max drawdown":None}

            df = yf.download(symbol, start=self.end_time - relativedelta(months=2), end=self.end_time, progress=False)

            pct_change = df["Adj Close"].pct_change().dropna()

            #Current Price
            stats["current price"] = df.iloc[-1, 4]

            #1 month high
            stats["high"] = df.iloc[-30:, 4].max()

            #1 month low 
            stats["low"] = df.iloc[-30:, 4].min()

            #measure volitility
            stats["standard deviation"] = self.compute_std(pct_change)

            #beta (volitility in relation to market)
            try:
                stats["beta"] = yf.Ticker(symbol).info["beta"]
            except Exception as e:
                stats["beta"] = "Not Available"

            #add number of price flips in period
            #percent_of_price_flips = pct_change.rolling(2).apply(lambda x: np.sign(x[0]) != np.sign(x[1])).value_counts()[1]/(len(pct_change)-1)
            stats["percent of flips"] = self.compute_percent_flips(pct_change)

            #measure skew of distribution
            stats["skew"] = self.compute_skew(pct_change)

            #30 day mean of % change
            stats["percent change mean"] = pct_change[-30:].mean()

            #previous day's % change
            stats["previous percent change"] = pct_change[-1]

            #sharpe ratio (annual)
            stats["sharpe ratio"] = ta.sharpe_ratio(df["Adj Close"])

            #measure kurtosis of distribution
            stats["kurtosis"] = self.compute_kurt(pct_change)

            #max drawdown
            stats["max drawdown"] = ta.max_drawdown(df.iloc[-30:, 4])

            self.buy_tickers["stats"].append(stats)


    def rsi(self, adj_close, symbol):

        metrics = ta.rsi(adj_close)

        if metrics[-1] > 35 and metrics[-2] < 30:
            self.buy_tickers["Symbol"].append(symbol)
            self.buy_tickers["TI"].append("RSI")
            self.buy_tickers["metric"].append(metrics[-1])           

    def macd(self, adj_close, symbol):

        metrics = ta.macd(adj_close, fast=4, slow=12, signal=3)

        if metrics.iloc[-1, 0] > metrics.iloc[-1, 2] and metrics.iloc[-2, 0] < metrics.iloc[-2, 2] and metrics.iloc[-1,0] > 0:
            self.buy_tickers["Symbol"].append(symbol)
            self.buy_tickers["TI"].append("MACD")
            self.buy_tickers["metric"].append(metrics.iloc[-1, 0])

    def strategy(self):

        for symbol in self.tickers[:500]:

            if symbol not in self.current_tickers["Symbol"]:

                try:
                    df = yf.download(symbol, start=self.start_time, end=self.end_time, progress=False)
                    self.rsi(df["Adj Close"], symbol)
                    #self.macd(df["Adj Close"], symbol)

                    #pct_change = df["Adj Close"].pct_change().dropna()

                    #self.market_std.append(self.compute_std(pct_change))
                    #self.market_percent_flips.append(self.compute_percent_flips(pct_change))
                    #self.market_skew.append(self.compute_skew(pct_change))
                    #self.market_kurtosis.append(self.compute_kurt(pct_change))

                except Exception as e:
                    self.errors["Symbol"].append(symbol)
                    self.errors["Error"].append(e)
                    self.errors["Date"].append(self.end_time)

        #self.stats()

        #write errors to error log 
        pd.DataFrame.from_dict(self.errors).to_csv("error_log.csv", encoding='utf-8', index=False)
        

    #consolidate statistics and send email
    def email(self):

        #print(self.buy_tickers) #for backtesting 

        df = pd.DataFrame.from_dict(self.buy_tickers)

        df = df.sort_values("metric", ascending=False)

        df["metric"] = df["metric"].round(2)

        df["text data"] = df["Symbol"].str.cat(df[["TI", "metric"]].astype(str), sep=", ")

        #print(df["text data"])
        #print(df.dtypes)

        email = SendText()
        email.send_text(df["text data"].values)




entry_strategy = Entry_Strategy()
entry_strategy.strategy()
entry_strategy.email()