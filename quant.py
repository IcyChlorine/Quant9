# def initialize(context):
#     set_benchmark('000300.XSHG')
#     set_option('use_real_price',True)
#     set_option('order_volume_ratio', 1)
#     set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_today_commission=0,close_commission=0.0003,min_commission=5), type='stock')
#     g.security = '000300.XSHG'   # 可多加股票
#     g.M = 15   # 20日均线
#     g.N = 2.2    # N
    
# def handle_data(context,data):
#     sr = attribute_history(g.security,g.M)['close']
#     ma20 = sr.mean()                                 # 20日均线
#     up = ma20 + g.N * sr.std()
#     down = ma20 - g.N * sr.std()
#     q=attribute_history(g.security,1)['high'].mean()
#     v=attribute_history(g.security,1)['close'].mean()
#     p = get_current_data()[g.security].day_open      # 开盘价
#     amount = context.portfolio.positions[g.security].total_amount # 总持仓 。或者开仓均线去表示
#     cash = context.portfolio.available_cash                       # 可用资金
    
#     # 如果p小于down，且没有持仓，则 买入
#     # 如果p大于up，且有持仓，则 卖出
#     if  q> up and amount == 0:
#         order_value(g.security,cash) 
#     elif v< down and amount > 0:
#         order_target(g.security,0)
import pandas as pd
import numpy as np
import talib

def load_data():
    return pd.read_csv("SH000300.csv")

def get_signal(data: pd.DataFrame, do_short=False):
    N = 2.2
    sr = data['Close']
    ma20 = sr.rolling(20).mean()                                 # 20日均线
    std = sr.rolling(20).std()
    up = ma20 + N * std
    down = ma20 - N * std
    q=data['High']
    k=data['Close']
    v=data['Low']
    holds = np.zeros_like(sr.values)
    buy = q>up
    sell = k<down
    short = v<down
    for i in range(1, len(holds)):
        if buy[i]:
            holds[i] = 1
        elif short[i]:
            holds[i] = -1 if do_short else 0
        elif sell[i]:
            holds[i] = -1 if do_short else 0
        elif holds[i-1]<0:
            holds[i] = holds[i-1] * 0.9
        else:
            holds[i] = holds[i-1]
    return holds
    # print(np.mean((q>up).astype(np.float32).values))# buy
    # print((v<down))# sell
    # p = get_current_data()[g.security].day_open      # 开盘价
    # amount = context.portfolio.positions[g.security].total_amount # 总持仓 。或者开仓均线去表示
    # cash = context.portfolio.available_cash                       # 可用资金
    
    # # 如果p小于down，且没有持仓，则 买入
    # # 如果p大于up，且有持仓，则 卖出
    # if  q> up and amount == 0:
    #     order_value(g.security,cash) 
    # elif v< down and amount > 0:
    #     order_target(g.security,0)

data = load_data()
holds = get_signal(data, do_short=False)
holds_short = get_signal(data, do_short=True)
from utils import sim_trade2, plot_idx, calc_sharpe_ratio, show_rate_stat, calc_drawdown
capital = sim_trade2(data["Close"].values, holds, friction = True, friction_rate = 0.001, print_func = lambda x: None, date = None)
capital_short = sim_trade2(data["Close"].values, holds_short, friction = True, friction_rate = 0.001, print_func = lambda x: None, date = None)

from matplotlib import pyplot as plt
print(calc_sharpe_ratio(capital))
print(calc_sharpe_ratio(capital_short))
# print(calc_drawdown(capital))
# print(show_rate_stat(data['Close'].values, capital, period=250))
# plot_idx(data["Close"].values, capital, log=False)
plt.figure(figsize=(16,4))
plt.plot(data["Close"].values/data["Close"].values[0],label='market')
plt.plot(capital/capital[0],label='w/o short')
plt.plot(capital_short/capital_short[0],label='w short')
plt.grid(); plt.legend()
# plt.savefig("with_short.png")
plt.show()