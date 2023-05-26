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
from matplotlib import pyplot as plt
from utils import sim_trade2, plot_idx, calc_sharpe_ratio, show_rate_stat, calc_drawdown, add_line

def load_data():
    return pd.read_csv("SH000300.csv")

def baseline(plt, data: pd.DataFrame):
    N = 2.2
    sr = data['Close']
    ma20 = sr.rolling(20).mean()
    std = sr.rolling(20).std()
    up = ma20 + N * std
    down = ma20 - N * std
    q=data['High']
    k=data['Close']
    v=data['Low']
    holds = np.zeros_like(sr.values)
    buy = q>up
    sell = (k<down)
    short = (v<down)
    for i in range(1, len(holds)):
        if buy[i]:
            holds[i] = 1
        elif sell[i] or short[i]:
            holds[i] = 0
        else:
            holds[i] = holds[i-1]
    return holds

def signal(data: pd.DataFrame, period=20):
    def linear(series: pd.Series):
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
        slope = linear(series)
        x = np.array(range(len(series)), dtype=np.float32) * slope
        x -= np.mean(x)
        series = series - series.mean() - x
        return series.std()
    
    sr = data['Close']
    ma = sr.rolling(period).mean()
    std = sr.rolling(period).std()
    lin = sr.rolling(period).apply(linear)
    no_lin = sr.rolling(period).apply(no_lin_std)
    return ma, std, lin, no_lin


def get_signal2(plt, data: pd.DataFrame, do_short=False):
    # sharp = 0.75
    N = 2.1
    sr = data['Close']

    ma40, std40, lin40, no_lin40 = signal(data, 40)
    ma20, std20, lin20, no_lin20 = signal(data, 20)
    ma10, std10, lin10, no_lin10 = signal(data, 10)

    # up20 = ma20 + N * no_lin20 + lin20 * 10
    up20 = ma20 + N * no_lin20 + lin20.apply(lambda x:x if x>0 else 0) * 10
    # up20 = ma20 + N * std20# + lin20.apply(lambda x:x if x>0 else 0) * 10
    down20 = ma20 - N * no_lin20 + lin40 * 10
    # down20 = ma20 - N * sr.rolling(20).std()
    add_line(plt, up20.fillna(sr[0]), normalize=True, label="up20")
    add_line(plt, down20.fillna(sr[0]), normalize=True, label="down20")
    buy = data['High'] > up20
    sell = data['Low'] < down20

    holds = np.zeros_like(sr.values)

    for i in range(1, len(holds)):
        h = holds[i-1]
        if buy[i]:
            holds[i] = 1
        elif sell[i]:
            holds[i] = 0
        elif h < 0:
            # 做空头寸衰减
            holds[i] = 0.95 * h
        else:
            # 持仓
            holds[i] = h
    add_line(plt, holds.astype(np.int8), label="holds")
    return holds

def get_signal3(plt, data: pd.DataFrame, do_short=False):
    # SOTA!
    N = 2.1
    sr = data['Close']

    ma80, std80, lin80, no_lin80 = signal(data, 80)
    ma40, std40, lin40, no_lin40 = signal(data, 40)
    ma20, std20, lin20, no_lin20 = signal(data, 20)
    ma10, std10, lin10, no_lin10 = signal(data, 10)

    # up20 = ma20 + N * no_lin20 + lin20 * 10
    up20 = ma20 + N * no_lin20 + lin20.apply(lambda x:x if x>0 else 0) * 10
    # up20 = ma20 + N * std20# + lin20.apply(lambda x:x if x>0 else 0) * 10
    down20 = ma20 - N * no_lin20 + lin40 * 10
    # down20 = ma20 - N * sr.rolling(20).std()
    # add_line(plt, up20.fillna(sr[0]), normalize=True, label="up20")
    # add_line(plt, down20.fillna(sr[0]), normalize=True, label="down20")
    buy = data['High'] > up20
    sell = data['Low'] < down20

    holds = pd.Series(np.zeros_like(sr.values))
    for i in range(1, len(holds)):
        h = holds[i-1]
        if buy[i]:
            holds[i] = 1
        elif sell[i]:
            holds[i] = 0
        elif h < 0:
            # 做空头寸衰减
            holds[i] = 0.95 * h
        else:
            # 持仓
            holds[i] = h
    add_line(plt, holds, label="holds")

    mean_holds = holds.rolling(80).mean()

    M = 1.4
    P = 1.2
    _up20 = ma20 + 3.0 * no_lin20 + lin40.apply(lambda x:x if x<0 else -x) * 10
    # _up20 = ma20 + M * std20# + lin40 * 10
    # up20 = ma20 + N * std20# + lin20.apply(lambda x:x if x>0 else 0) * 10
    _down20 = ma20 - M * std20 - P * std40 - lin40.apply(lambda x:x if x<0 else -x) * 10 - (lin80 - lin40).apply(lambda x:x if x>0 else 0)
    # down20 = ma20 - N * sr.rolling(20).std()
    __down20 = ma20 - M * std20 - P * std40
    add_line(plt, _up20.fillna(sr[0]), normalize=True, label="up20")
    add_line(plt, __down20.fillna(sr[0]), normalize=True, label="ma40+M*no_lin40")
    add_line(plt, _down20.fillna(sr[0]), normalize=True, label="down20")

    short = data['Close'] < _down20
    run = data['Close'] > _up20

    short_holds = np.zeros_like(sr.values)

    for i in range(1, len(holds)):
        h = short_holds[i-1]
        # 检验是否存在趋势
        if short[i]:
            short_holds[i] = -1
        elif run[i]:
            short_holds[i] = 0
        elif h < 0:
            # 做空头寸衰减
            short_holds[i] = 0.999 * h
        else:
            # 持仓
            short_holds[i] = h
    add_line(plt, short_holds, label="short_holds")
    return short_holds + holds


data = load_data()
plt.figure(figsize=(16,4))
add_line(plt, data['Close'].values, normalize=True, label='close')
# add_line(plt, data['High'].values, normalize=True, label='high')
# add_line(plt, data['Low'].values, normalize=True, label='low')

# baseline
baseline_holds = baseline(plt, data)
baseline_capital = sim_trade2(data["Close"].values, baseline_holds, friction = True, friction_rate = 0.001, print_func = lambda x: None, date = None)
add_line(plt, baseline_capital, normalize=True, label='baseline')
print(f"baseline sharp: {calc_sharpe_ratio(baseline_capital)}")

# ours
holds = get_signal3(plt, data)
capital = sim_trade2(data["Close"].values, holds, friction = True, friction_rate = 0.001, print_func = lambda x: None, date = None)
add_line(plt, capital, normalize=True, label='ours')
print(f"our sharp: {calc_sharpe_ratio(capital)}")
show_rate_stat(data['Close'].values, capital, 250)
plt.grid(); plt.legend()
# plt.savefig("with_short.png")
plt.show()