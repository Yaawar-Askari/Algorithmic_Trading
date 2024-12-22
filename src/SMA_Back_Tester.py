#!/usr/bin/env python
# coding: utf-8
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class SMA_Back_Tester():
    def __init__(self,symbol,SMA_S,SMA_L,start,end):
        self.symbol=symbol
        self.SMA_S=SMA_S
        self.SMA_L=SMA_L
        self.start=start
        self.end=end
        self.results=None
        self.get_data()
        
    def get_data(self):
        df=yf.download(self.symbol,start=self.start, end=self.end)
        data=df.Close.to_frame()
        data["returns"]=np.log(data.Close.div(data.Close.shift(1)))
        data["SMA_S"]=data.Close.rolling(self.SMA_S).mean()
        data["SMA_L"]=data.Close.rolling(self.SMA_L).mean()
        data.dropna(inplace=True)
        self.data2=data
        
        return data
   
    def test_results(self):
        data = self.data2.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["returns"] * data["position"].shift(1)
        data.dropna(inplace=True)
        data["returnsbh"] = data["returns"].cumsum().apply(np.exp)
        data["returnstrategy"] = data["strategy"].cumsum().apply(np.exp)
        perf = data["returnstrategy"].iloc[-1]
        outperf = perf - data["returnsbh"].iloc[-1]

        ret = np.exp(data["strategy"].sum())
        std = data["strategy"].std() * np.sqrt(252)

        self.results = (round(perf, 6), round(outperf, 6))

        # Store the cumulative returns in self.data2 for plotting
        self.data2["returnsbh"] = data["returnsbh"]
        self.data2["returnstrategy"] = data["returnstrategy"]

        return self.results    
    
    def plot_results(self):
        if self.results is None:
            print("Run the test please")
        else:
            title = "{} | SMA_S={} | SMA_L={}".format(self.symbol, self.SMA_S, self.SMA_L)
            returnsbh, returnstrategy = self.results  # Unpack the tuple
            data_to_plot = pd.DataFrame({'returnsbh': [returnsbh], 'returnstrategy': [returnstrategy]})

            # Concatenate with existing DataFrame and plot
            data_to_plot = pd.concat([self.data2[['returnsbh', 'returnstrategy']], data_to_plot])
            data_to_plot.plot(title=title, figsize=(12, 8))

            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.show()


