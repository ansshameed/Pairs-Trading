import yfinance as yf #yahoo finance
import pandas as pd 
import numpy as np 
from ta.momentum import RSIIndicator #technical analysis library
from ta.volatility import BollingerBands 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller