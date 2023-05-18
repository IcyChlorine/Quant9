import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import talib
from datetime import datetime

# plot
def plot_idx(market_vector, capital_vector, log = True, downsample = 1):
	plt.figure(figsize=(16,4))
	if log:
		plt.plot(np.log(market_vector[::downsample]/market_vector[0]),label='market')
		plt.plot(np.log(capital_vector[::downsample]/capital_vector[0]),label='ours')
	else:
		plt.plot(market_vector[::downsample]/market_vector[0],label='market')
		plt.plot(capital_vector[::downsample]/capital_vector[0],label='ours')
	plt.grid(); plt.legend()

def plot_rate(market_vector, capital_vector, log = True, downsample = 1):
	plt.figure(figsize=(16,4))
	X = capital_vector/market_vector
	X = X/X[0]
	if log: X = np.log(X)
	plt.plot(X[::downsample],color='C1')
	plt.grid(); plt.legend()

# calculate yield rate
def calc_rate(price_vec, nr_cycles):
	tot_cycles = len(price_vec)
	rate = price_vec[-1]/price_vec[0]
	return np.exp( np.log(rate)*(nr_cycles/tot_cycles) ) - 1
def calc_alpha(market_vec, capital_vec, nr_cycles):
	return calc_rate(capital_vec, nr_cycles) - calc_rate(market_vec, nr_cycles)

def show_rate_stat(Market: np.ndarray, Capital: np.ndarray, period = 24*60):
	market_rate = calc_rate(Market, period)
	trade_rate  = calc_rate(Capital, period)
	print(f'Market   growth rate: {100*market_rate:6.2f}%(per period)   {100*(Market[-1]/Market[0]-1):6.2f}%(total)')
	print(f'Strategy growth rate: {100*trade_rate:6.2f}%(per period)   {100*(Capital[-1]/Capital[0]-1):6.2f}%(total)')
	print(f'Extra    growth rate: {100*(trade_rate-market_rate):6.2f}%(per period)   {100*(Capital[-1]/Capital[0] - Market[-1]/Market[0]):6.2f}%(total)')

def filter_repeated_signal(signal: np.ndarray):
	'''保证买卖点信号一一对应，过滤掉连续的买点和卖点信号'''
	holding_share = False
	for i in range(len(signal)):
		if holding_share:
			if   signal[i] == 1: signal[i] = 0
			elif signal[i] ==-1: holding_share = False 
		else:
			if   signal[i] ==-1: signal[i] = 0
			elif signal[i] == 1: holding_share = True
	return signal

# timing strategies: MACD
def calc_MACD_timing(market_vector, param = (12, 26, 9)):
	MACD, Signal, _ = talib.MACD(np.log(market_vector), fastperiod=param[0], slowperiod=param[1], signalperiod=param[2])
	Signal_Cross = np.zeros(shape = (len(MACD),), dtype = np.int32)
	for i in range(1, len(market_vector)):
		if MACD[i] > Signal[i] and MACD[i-1] <= Signal[i-1]:
			Signal_Cross[i] = 1
		elif MACD[i] < Signal[i] and MACD[i-1] >= Signal[i-1]:
			Signal_Cross[i] = -1
	return Signal_Cross

def calc_MACD_timing2(market_vector, param = (12, 27, 52, 0.0002, 7)):
	'''the BEST timing strategy! Note that the parameters have been TUNED
	and their numerical values are very important!'''
	X = np.log(market_vector)
	MACD, Signal, _ = talib.MACD(X, fastperiod=param[0], slowperiod=param[1], signalperiod=param[2])
	
	MACD_diff = MACD.copy(); 
	MACD_diff[1:] = MACD_diff[1:]-MACD_diff[:-1]
	MACD_diff[0] = np.nan
	threshold = param[3]
	AVG = talib.SMA(X, timeperiod=param[4])
	Signal_Cross = np.zeros(shape = (len(MACD),), dtype = np.int32)

	for i in range(param[4], len(market_vector)):
		if np.abs(MACD_diff[i])>threshold and MACD[i] > Signal[i] and MACD[i-1] <= Signal[i-1]:
			Signal_Cross[i] = 1
		elif AVG[i]<=AVG[i-1] and X[i] < AVG[i] and X[i-1] >= AVG[i-1]:
			Signal_Cross[i] = -1
	return Signal_Cross

def calc_trading_rate(SignalCross, nr_cycles):
	'''calculate trade frequency per nr_cycles'''
	return np.sum(np.abs(SignalCross)) / (len(SignalCross)/nr_cycles)


def format_date(date_int) -> str:
	dt_obj = datetime.strptime(str(date_int),'%Y%m%d')
	return datetime.strftime(dt_obj, '%Y-%m-%d')

def sim_trade(price, signal, friction = False, friction_rate = 0.001, print_trade_log = False, date = None):
	'''根据大盘信息和已经计算好的买卖信号，进行模拟交易
	friction 表示是否考虑佣金损耗'''

	# 计算收益率和资金曲线
	fric_decay_ratio = 1 # 买卖时交易佣金带来的损耗
	if friction: fric_decay_ratio = 1 - 2*friction_rate
	cash = 1000000.0  # 起始资金
	shares = np.zeros(shape = price.shape, dtype = np.float32) # 股份
	capital = np.zeros(shape = price.shape, dtype = np.float32) # 总资产
	capital[0] = cash

	for i in range(1, len(price)):
		if print_trade_log: date_str = format_date(date[i])
		if signal[i] == +1: # buy
			shares[i] = shares[i-1] + cash / price[i]
			if print_trade_log:
				print(f'{date_str}: BUY  at open price: {price[i]:5.2f}')
				print(f'            Get {shares[i]:.2f} shares with cash {cash:.2f}')
			cash = 0
		elif signal[i] == -1: # sell
			cash += price[i] * shares[i-1] * fric_decay_ratio
			if print_trade_log:
				print(f'{date_str}: SELL  at open price: {price[i]:5.2f}')
				print(f'            Get {cash:.2f} cash with shares {shares[i-1]:.2f}')
			shares[i] = 0
		else:
			shares[i] = shares[i-1]
		
		capital[i] = cash + shares[i]*price[i]

	return capital