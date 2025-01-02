import yfinance as yf #yahoo finance
import pandas as pd 
import numpy as np 
from ta.momentum import RSIIndicator #technical analysis library
from ta.volatility import BollingerBands 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller

'''NOTES 

- Pair = When difference between the two assets mean revert or are cointegrated. Criteria: 
-> Cointegration: Two assets must be cointegrated (p-value < 0.05) 
-> Stationarity Test: Linear combo (spread) of two assets must be stationary (p < 0.05) 
-> Stable Mean and Variance: Spread should oscillate around stable mean and relatively constant variance 
-> Economic/Logical Relationship: Economic link (e.g. competitors in same industry)

COINTEGRATION CHECK 
- Cointegration = Two or more stationary time series (e.g. stock prices) have long-term equilibrium relationship 
-> s.t. linear combination (spread) is stationary, implying deviations between them tend to revert to 
-> a mean overtime. Expect price difference between the pair to come back to common long-run mean

STATIONARY CHECK 
Stationary time series = statistical properties e.g. mean, variance; remain constant overtime
-> ADF Check = checks if time series is stationary by checking presence of a unit root
--> Unit root = time-series is not stationary; mean and variance change overtime and tendency to drift rather than revert to fixed mean
-> both time series do not need to be stationary individually, but their linear combination (spread) must be stationary
-> Each asset price time series could be non-stionary but the price difference (spread) between the pair should be stationary

Interpretation of ADF Results
- t-statistic: how far series deviates from mean-reverting process. Low t-test stat (more negative) suggests stronger evidence for cointegration 
- critical values: test checks if t-stat is below critical thresholds (1%, 5%, 10%). If not, you cannot reject (must accept) the null hypothesis of "no cointegration"
- p-value: probability of observing the test statistic under null hypothesis. P-value < 0.05 is cutoff for cointegration (if p < 0.05 cointegrated)

Log Price Data Conversion
- Highlights proportional relationships and stabilises variance
- Can reveal cointegraton in series with different price scales or growth rates

Long backtest (24 years) is okay for cointegration check 
Optimal backtest period for actual PTS backtest (3-5 years) to exploit mean reversion 

Window: 
- no. of periods (e.g. days, hours) to calculate MA or other rolling statistic 
-> e.g. BB with window of 20 uses the last 20 periods of data to compute MA and SD. 
-> window size determines how sensitive the indicator is to recent daa, the smaller windows being more responsive and larger windows smoothing out fluctuations
- General period: corresponds to frequency of data. e.g. if data is daily prices the window=14 is 14 days

Bollinger Bands
- 3 lines: SMA  (middle), upper band (SMA + 2SD), lower band (SMA - 2SD)

RSI (Relative Strength Indicator)
- Measures speed and magnitude of price changes to identify: 
- Overbought (above 70) and Oversold (below 30) 

Z-score: 
- Measures how far spread is from its rolling mean (in terms of standard deviations) over the specified periods 
- Identifies extreme deviations from the mean 
- Signals potential trading opportunities
- Calculations: (Current Spread - Rolling Mean of Spread / Rolling SD
--> How many standard deviations the apread is from its mean 
- Rolling = Moving/sliding calculation for specific window/consecutive datapoints; updating when new data added and old data is dropped

Strat assumes that if Stock 1 dips below the lower band BB (oversold) and expected to revert upwards then: 
- Stock 2 (being cointegrated) may move in the opposite direction or stay relatively stable
- This is because we are going based off the info about spread. If spreads dip then assumes 1 stock doing well and the other one doing poorly
- If the spread is very low in RSI and Bands then it assumes stock 1 is underperforming (long) and stock 2 overperforming (short)
- Shorting Stock 2 hedges against overall market movements, isolating the profit potential from the pair's mean-reversion relationship

Portfolio update (everyday): 
- Includes remaining capital (either 0 once bought or if sold then all long positions (sold) - short positions (bought back)) 
- Remaining capital + value of long position (long shares * stock 1 current price) - short position (short shares * stock 2 current price)

Returns list: 
- Calculates daily percentage change of portfolio everyday. 
- If loss then return will be negative

Total Return Metric: 
- Percentage change between portfolio value at end of backtest and the capital given at the start

Sharpe Ratio Metric: 
- Risk-adjusted return of portfolio. How much excess return (over risk-free rate) is earned per unit of risk (volatility)
-> Returns = everyday percentage change of portfolio compared to previous day (daily returns)
- (Mean Portfolio Returns / STD Portfolio Returns) * Sqrt(Annualisation Factor)
- np.mean(returns) = average daily return of portoflio. Expected return for single trading day in backtest 
- np.std(returns) = average STD of daily returns, measure of portfolio's volatility. Higher volatility means higher risk 
- Division = average daily return of portfolio / portfolio volatility
- Risk-adjusted return: SR uses ratio of average return to volatility to express how much return is being earned relative to the risk taken (volatility)
- Annualisation Factor: SR usually expressed as annualised basis, daily returns are converted into annualised returns 
-> approx. 252 days in traidng year. Sqrt of trading days in a year
- Higher SR = better risk-adjusted performance 
-> Above 1 is considered good, anything below 1 means portfolio might not be providing adequate returns for its risk level 
- Usually you do (mean portfolio returns - risk free rate) / std returns 
-> Risk-free rate = return on investment with 0 risk e.g. yields on US Treasury Bills
-> Risk-free rate omitted from here for simplicity (assuming its 0)
-> Assumes all trades done under risk as calculation evaluates return achieved relative to volatility (risk of those returns), without offsetting for any guaranteed or risk-free component

Maximum Drawdown Metrics: 
- Max Drawdown: max observed loss from portfolio's peak value to its lowest point before recovering, as a percentage, during specific time period
- 1 - (portfolio value/peak value)
- Finds max largest drawdown observed in backtest
- 1 - is used to express drawdown as a proportion or percentage of the peak value. Value between 0 and 1
- Division calcilates portfolio's value as fraction of its highest value up to that point (current peak). 
- Calculates max drawdown of everyday and finds maximum 
'''

def fetch_data(symbols, start_date, end_date): 
    #Fetch data for symbols
    data = {} #Stores data as dictionary
    for symbol in symbols: #loops through symbols in symbols list
        df = yf.download(symbol, start=start_date, end=end_date) #downloads data for symbol (start to end date)
        if not df.empty: #check if data is valid; data exists
            data[symbol] = df['Close'] #get daily close prices
        else: 
            print(f"Failed to download data for {symbol}") 

    #Combine data into single DataFrame 
    if data: 
        price_data = pd.concat(data, axis=1) #aligns data by index (dates)
        price_data.columns = symbols #rename columns to stock symbols

        #Log transformation - not sure if needed? 
        price_data = np.log(price_data) 
        print(price_data.tail())
        
        #Save data to CSV file 
        price_data.to_csv("backtest_data.csv") 
        print("Data saved to backtest_data.csv")
        return price_data

    else: 
        print("No valid data fetched.")
        return None

#Check if the two stocks time series (stock prices) have cointegrated relationship 
#Means their linear combination (spread) is stationary; stationary spread implies deviations form the mean are temporary and revert overtime
def test_cointegration(price_data, symbols):
    print("\nCointegration Test Results:")
    # Extract price series
    stock1 = price_data[symbols[0]]
    stock2 = price_data[symbols[1]]

    # Perform cointegration test
    t_stat, p_value, critical_values = coint(stock1, stock2)
    #print(f"Cointegration test results for {symbols[0]} and {symbols[1]}:")
    #print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")
    #print(f"Critical Values: {critical_values}")

    # Interpret results
    if p_value < 0.05:
        print(f"{symbols[0]} and {symbols[1]} are cointegrated (p-value < 0.05).")
        return True
    else:
        print(f"{symbols[0]} and {symbols[1]} are not cointegrated (p-value >= 0.05).")
        return False

#Augmented Dickey-Fuller (ADF) test on individual time series
def adf_test(price_data, symbols):
    print("\nADF Test Results:")
    
    #Test stationarity of the spread (linear combination of time series) 
    spread = price_data[symbols[0]] - price_data[symbols[1]]
    adf_stat, p_value, _, _, critical_values, _ = adfuller(spread) #adf test on spread
    #print("\nADF Test for Spread:")
    #print(f" ADF Statistic: {adf_stat}")
    print(f"p-value: {p_value}")
    #print(f" Critical Values: {critical_values}")
    if p_value < 0.05:
        print("The spread is stationary (p-value < 0.05).")
        return True
    else:
        print("The spread is not stationary (p-value >= 0.05).")
        return False

def calculate_indicators(spread, window=20): 

    #spread[]: adding new columns to Spreads dataframe (indicator information for each day along)
    #Bollinger Bands - SMA and upper/lower bands. 
    #1st param = numeric series (spread data of closing prices) and window param (period over which MA and STD calculated)
    #Upper and Lower band updates begin on day 20 to allow enough data for the window
    bollinger = BollingerBands(spread['Spread'], window=window) #20 day Bollinger Band.
    spread['BB_upper'] = bollinger.bollinger_hband() #upper band 
    spread['BB_lower'] = bollinger.bollinger_lband() #lower band 
