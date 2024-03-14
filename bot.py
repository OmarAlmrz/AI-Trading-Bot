from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import date
from timedelta import Timedelta 

API_KEY = "PKJB8CY35VX79VER7N1G" 
API_SECRET = "rno6jBwnLMe7bREMCG7CgQg4f6Pu9aHRqMlcel1b" 
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}

# Load the model and tokenizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Sigma/financial-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("Sigma/financial-sentiment-analysis").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    """
    This function takes in a list of news headlines and returns the sentiment of the news.
    If no news is provided, the function returns a neutral sentiment.
    """
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)
        # This will return the logits of all the news headlines in term of the sentiment (positive, negative, neutral)
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"] 
        # We sum the logits of all the news headlines and apply a softmax to get the probability of the sentiment
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        # We return the probability of the sentiment with the highest probability
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]

class MLTrader(Strategy): 
    def initialize(self, symbol:str="NVDA", cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self): 
        """ Calculate the position size based on the cash at risk and the last price of the stock.

        Returns:
            cash: float
                The amount of cash in the account
            last_price: float
                The last price of the stock
            quantity: float
                The number of shares to buy or sell
        """
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self): 
        """Get the current date and the date three days prior to the current date."""
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        """Get all the news for the stock of the last three days and estimate the sentiment of the news."""
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 

     
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing() 
        probability, sentiment = self.get_sentiment()

        # If the cash is greater than the last price, we can buy the stock
        if cash > last_price: 
            # If the sentiment is positive and the probability is greater than .999, we buy the stock
            if sentiment == "positive" and probability > .999: 
                if self.last_trade == "sell": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "buy", 
                    take_profit_price=last_price*1.20, 
                    stop_loss_price=last_price*.95
                )
                self.submit_order(order) 
                self.last_trade = "buy"
                
            # If the sentiment is negative and the probability is greater than .999, we sell the stock
            elif sentiment == "negative" and probability > .999: 
                if self.last_trade == "buy": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell", 
                    take_profit_price=last_price*.8, 
                    stop_loss_price=last_price*1.05
                )
                self.submit_order(order) 
                self.last_trade = "sell"

start_date = datetime(2020,1,1)
end_date = datetime(2023,12,31) 
broker = Alpaca(ALPACA_CREDS) 
strategy_params = parameters={"symbol":"NVDA", "cash_at_risk":.5}

strategy = MLTrader(broker=broker, parameters=strategy_params)
strategy.backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, 
    parameters=strategy_params
)

# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
