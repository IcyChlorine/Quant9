import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import talib
from datetime import datetime

def add_line(plt, vector, normalize=False, log=False, label=None, **kwargs):
    v = vector.copy()
    if normalize: v = v/v[0]
    if log: v = np.log(v)

    plt.plot(v, label=label, **kwargs)

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

def noob_shares_management(price, signal, no_short=False):
    '''
    输入一个MACD等策略的信号, 将普通的信号转换为持仓信号.
    总是满仓做空或做多
    '''
    holding_rate = np.zeros_like(signal, dtype=np.float32)
    for i in range(len(signal)):
        if signal[i] > 0:
            holding_rate[i] = 1
        elif signal[i] < 0:
            holding_rate[i] = -1 if not no_short else 0
        else:
            holding_rate[i] = holding_rate[i-1] if i>=1 else 0.
    return holding_rate

def sim_trade2(price, signal, friction = False, friction_rate = 0.001, print_func = lambda x: None, date = None):
    '''根据大盘信息和已经计算好的买卖信号，进行模拟交易
    signal 为 [-1, 1] 内的实数, 表示看空/看多的强度. 当 signal=1 时全仓做多, signal=-1 时做空
    signal 需持续保持信号值以持续持仓(例如连续的10个1.0表示连续10天保持满仓). 因此 MACD 等策略得出的信号还需经过一次仓位管理的过程才能输入
    friction 表示是否考虑佣金损耗'''
    
    # 计算收益率和资金曲线
    if not friction: friction_rate = 0
    fric_decay_ratio = 1 - 2*friction_rate # 买卖时交易佣金带来的损耗
    cash = 1000000.0  # 起始资金
    shares = np.zeros(shape = price.shape, dtype = np.float32) # 股份
    capital = np.zeros(shape = price.shape, dtype = np.float32) # 总资产
    capital[0] = cash
    assert np.all(abs(signal) <= 1)

    for i in range(1, len(price)):
        if date is not None:
            date_str = format_date(date[i])
        else:
            date_str = f"{i:4.0f}"
        # 计算账户等值资产(不考虑佣金)
        # 如果 shares 小于 0 表示有空头头寸
        _cash = cash + shares[i-1] * price[i]
        # 目标持仓
        # 随着 signal 在 -1 和 1 之间变化
        # 可以表达全仓做空(借入和当前资本等值的股票做空)和全仓做多之间的所有情况
        _shares = _cash * signal[i] / price[i]

        if _shares < shares[i-1]:
            # 旧的持仓高于新持仓
            shares[i] = _shares
            cash = cash + (shares[i-1] - _shares) * price[i] * fric_decay_ratio
            print_func(f'{date_str}: SELL at open price: {price[i]:5.2f}')
            print_func(f'            Get {cash:.2f} cash with shares {shares[i-1]:.2f}')
        elif _shares > shares[i-1]:
            # 新的持仓高于原持仓
            shares[i] = _shares
            cash = cash - (_shares - shares[i-1]) * price[i]
            print_func(f'{date_str}: BUY  at open price: {price[i]:5.2f}')
            print_func(f'            Get {shares[i]:.2f} shares with cash {cash:.2f}')
        else:
            # 持仓不变
            shares[i] = shares[i-1]
        
        capital[i] = cash + shares[i]*price[i]

    return capital

def calc_drawdown(equity):
    '''计算最大回撤，返回[0,1]间的实数'''
    ans = 1
    high = equity[0]
    for i in range(len(equity)):
        if equity[i]>high: high=equity[i]
        else: ans = min(ans, equity[i]/high)
    return ans

def calc_sharpe_ratio(equity, 
                      nr_yearly_cycles=250, 
                      riskless_annual_yield=0.015, 
                      nr_samples=500, rd_seed=None):
    '''计算基于年化收益率的夏普比。nr_nearly_cycles: 一年有多少个交易周期。'''
    if rd_seed is not None: np.random.seed(rd_seed)
    if len(equity) < nr_yearly_cycles: raise ValueError("Series too short!")
    if nr_samples > len(equity): nr_samples = len(equity)

	# 标准差计算方法：每天的平均收益率的标准差 x sqrt(N)，N是一年中的交易周期数。
	# 基于一些统计理论，年化收益率的标准差 ~ 日收益率的标准差 x sqrt(N)。

    tot_yield = equity[-1]/equity[0] - 1
    yearly_yield = np.exp(np.log(tot_yield+1) * nr_yearly_cycles/len(equity)) - 1
    samples = [
        equity[t+nr_yearly_cycles]/equity[t] for t in range(len(equity)-nr_yearly_cycles)
    ]
    
    daily_return = np.log(equity[1:] / equity[:-1])
    std = np.std(daily_return) * np.sqrt(nr_yearly_cycles)
    # for i in range(nr_samples):
    #     t = np.random.randint(0, len(equity)-nr_yearly_cycles)
    #     samples.append(equity[t+nr_yearly_cycles]/equity[t])
    # std = np.array(samples).std()

    return (yearly_yield - riskless_annual_yield) / std

def show_advanced_stat(Market: np.ndarray, Capital: np.ndarray, period = 252):
    market_ratio = calc_sharpe_ratio(Market, nr_yearly_cycles = period)
    our_sharpe_ratio = calc_sharpe_ratio(Capital, nr_yearly_cycles = period)
    print(f'The sharpe ratio of the strategy is {our_sharpe_ratio:.3f} vs {market_ratio:.3f}(market)')
    
    max_drawdown = calc_drawdown(Capital)
    print(f'Max draw down is: {(1-max_drawdown)*100:.2f}%.')