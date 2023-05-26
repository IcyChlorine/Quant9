import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
from utils import *

def load_data():
    return pd.read_csv("SH000300.csv")

def baseline(plt, data: pd.DataFrame):
    # 计算bolling线
    N = 2.2
    sr = data['Close']
    ma20 = sr.rolling(20).mean()
    std20 = sr.rolling(20).std()
    up = ma20 + N * std20
    down = ma20 - N * std20
    
    high=data['High']
    close=data['Close']
    low=data['Low']
    holds = np.zeros_like(sr.values)
    buy = high>up
    sell = (close<down)
    short = (low<down)
    
	# 当天最高价高于上轨则买入，当天最低价或收盘价低于下轨则卖出
    for i in range(1, len(holds)):
        if buy[i]:
            holds[i] = 1
        elif sell[i] or short[i]:
            holds[i] = 0
        else:
            holds[i] = holds[i-1]
    return holds

def magic_bolling(data: pd.DataFrame, period=20):
    def linear(series: pd.Series):
        '''根据最小二乘法计算一个序列的系数，自变量为range(0,len(series))'''
        n = len(series)
        x = np.array(range(n), dtype=np.float32)
        x -= np.mean(x)
        # 计算自变量的均值和因变量的均值
        y_mean = np.mean(series)
        # 计算自变量和因变量的差值乘积之和以及自变量的差值平方和
        xy_diff_sum = np.sum(x * (series - y_mean))
        x_diff_squared_sum = np.sum(x ** 2)
        # 计算回归系数
        slope = xy_diff_sum / x_diff_squared_sum
        return slope

    def no_lin_std(series: pd.Series):
        '''计算一个序列扣除线性分量后，残差的标准差'''
        slope = linear(series)
        lin_component = np.array(range(len(series)), dtype=np.float32) * slope
        lin_component -= np.mean(lin_component)
        residue = series - series.mean() - lin_component
        return residue.std()
    
    close = data['Close']
    ma     = close.rolling(period).mean()
    std    = close.rolling(period).std()
    lin    = close.rolling(period).apply(linear)
    rstd   = close.rolling(period).apply(no_lin_std)
    # rstd = residue std
    return ma, std, lin, rstd


def calc_mbolling_csignal(plt, data: pd.DataFrame, do_short=False):
    '''计算基于magic bolling的持仓信号。csignal - continuous signal - 表示每日持续持仓信号
    do_short表示做空与否'''
    # sharpe = 0.76
    N = 2.1
    close = data['Close']

	# calculate out custom bolling bands - MAGIC BOLLING
    ma40, std40, lin40, rstd40 = magic_bolling(data, 40)
    ma20, std20, lin20, rstd20 = magic_bolling(data, 20)
    #ma10, std10, lin10, rstd10 = magic_bolling(data, 10)

    up20   = ma20 + N * rstd20 + 10 * lin20 * (lin20>0)
    down20 = ma20 - N * rstd20 + 10 * lin40
    rup20  = ma20 + N * rstd20 + 10 * lin40
    rdown20= ma20 - N * rstd20 + 10 * lin20 * (lin20<0)
	
    #add_line(plt,   up20.fillna(close[0]), normalize=True, label="up20")
    #add_line(plt, down20.fillna(close[0]), normalize=True, label="down20")
    buy  = data['High'] > up20
    sell = data['Low' ] < down20
    rbuy = data['Low' ] < rdown20
    #rsell= data['High'] > rup20

    holds = np.zeros_like(close.values)

    for i in range(1, len(holds)):
        h = holds[i-1]
        if buy[i]:
            holds[i] = 1
        elif sell[i]:
            holds[i] = 0
            if do_short and rbuy[i]:
                holds[i] = -1
        elif do_short and rbuy[i]:
            holds[i] = -1
        elif h < 0:
            # 做空头寸衰减
            holds[i] = 0.95 * h
        else:
            # 持仓
            holds[i] = h
    #add_line(plt, holds.astype(np.int8), label="holds")
    return holds


data = load_data()
plt.figure(figsize=(16,4))
add_line(plt, data['Close'].values, normalize=True, label='market')
# add_line(plt, data['High'].values, normalize=True, label='high')
# add_line(plt, data['Low'].values, normalize=True, label='low')

# ================================================================= #
# baseline - strategy based on traditional bolling bands
# ================================================================= #
baseline_holds = baseline(plt, data)
baseline_capital = sim_trade2(data["Close"].values, baseline_holds, 
                              friction = True, friction_rate = 0.001, 
                              print_func = lambda x: None, date = None)
add_line(plt, baseline_capital, normalize=True, label='simple bolling strategy', linestyle='--')
print('---------------Baseline (Simple Bolling Strategy)-------------------')
print('------------------------<----annual----->----<-2010-2023->----------')
show_rate_stat(data['Close'].values, baseline_capital, period = 252)
show_advanced_stat(data['Close'].values, baseline_capital, period = 252)
print('')


# ================================================================= #
# our strategy - strategy based on custom magic bolling bands(仅做多)
# ================================================================= #
print('Running Strategy...(it may take a few seconds)......')
holds = calc_mbolling_csignal(plt, data)
capital = sim_trade2(data["Close"].values, holds, 
    				 friction = True, friction_rate = 0.001, 
    				 print_func = lambda x: None, date = None)

add_line(plt, capital, normalize=True, label='(ours)magic bolling strategy(仅做多)', color='C5')

print('-----------------Magic Bolling Strateg(仅做多)----------------------')
print('------------------------<----annual----->----<-2010-2023->----------')
show_rate_stat(data['Close'].values, capital, period = 252)
show_advanced_stat(data['Close'].values, capital, period = 252)
print('')


# ================================================================= #
# ours strategy - strategy based on custom magic bolling bands(+做空)
# ================================================================= #
print('Running Strategy...(it may take a few seconds)......')
# ours - strategy based on CUSTOM bolling bands - 带做空
holds = calc_mbolling_csignal(plt, data, do_short=True)
capital = sim_trade2(data["Close"].values, holds, 
    				 friction = True, friction_rate = 0.001, 
    				 print_func = lambda x: None, date = None)

add_line(plt, capital, normalize=True, label='(ours)magic bolling strategy(+做空)', color='C3')

print('------------------Magic Bolling Strateg(做多+做空)------------------')
print('------------------------<----annual----->----<-2010-2023->----------')
show_rate_stat(data['Close'].values, capital, period = 252)
show_advanced_stat(data['Close'].values, capital, period = 252)
print('')

# ======================== #
# End of strategies
# ======================== #

plt.grid(); plt.legend()
#plt.savefig("strategy.png")
plt.show()