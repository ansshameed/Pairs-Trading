# **Pairs-Trading Strategy (PTS)**

## **What is the Pairs Trading Strategy?**
- Pairs Trading is a **market-neutral strategy** to identify two historically correlated assets
- Takes **long position in underperforming asset** and **short position in outperforming asset** when price relationship diverges
- Profits made when **prices revert to mean**
- Demonstrates the potential to **exploit statistical arbitrage opportunities while minimising market risk**

## **Objective**

- Quantitative PTS to identify mean-reverting relationships between two assets, leveraging **statistical and technical analysis**

## **Key Features** 
- **Data Acquisition**:
  - `yfinance` to fetch historical data for backtesting. 
  - Backtesting used for both identifying 'pairs' of assets and for strategy testing
  - `backtest_data_diff.csv`: Daily data (12/2019-12/2024) of daily asset spread
  - `backtest_data.csv`: Daily data (12/2000 - 12/2024) of daily asset prices (to identify valid pair)
- **Pairs Validation**: Verifying two assets are a **pair**
  - **Cointegration Test**: Validates long-term equilibrium (p-value < 0.05)
  - **Stationarity Test**: Confirms mean-reverting spread using **ADF** (Augmented Dickey-Fuller) test (p-value < 0.05)
- **Technical Indicators**:
  - **Bollinger Bands**: Identifies overbought/oversold conditions over 20-day rolling window
  - **RSI** (Relative Strength Indicator): Identifies momentum over 14-day rolling window (oversold < 30, overbought > 70)
  - **Z-Score**: Measures spread deviations from the mean.
- **Trading Signals**:
  - **Buy**: Spread < Lower Bollinger Band, RSI < 30, Z-Score < -2
  - **Sell**: Spread > Upper Bollinger Band, RSI > 70, Z-Score > 2
- **Performance Metrics**:
  - **Total Return**: Portfolio growth percentage
  - **Sharpe Ratio**: Risk-adjusted return
  - **Max Drawdowm**: Largest Portfolio Decline
- **Visualisations**:
  - Spread with BB
  - Price time series of paired assets
  - Portfolio value evolution

## **Dependencies**
- ```yfinance```
- ```pandas```
- ```numpy```
- ```matplotlib```
- ```ta```
- ```statsmodel```

## **Sample Screenshots**

### **Bollinger Band Spread** 

![Screenshot 2025-01-02 at 18 12 34](https://github.com/user-attachments/assets/173fd87c-5d0d-4019-9888-a885449d1f1c)

### **Price Time Series of Asset Pair (PEP and KO)** 

<img width="960" alt="Screenshot 2025-01-02 at 18 13 25" src="https://github.com/user-attachments/assets/1eb5329c-dd2e-4fd0-b882-715d6e634d38" />



  
