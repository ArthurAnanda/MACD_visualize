import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('expand_frame_repr', False)
pd.set_option('max_rows', 100)


def EMA(tsPrice, period=5, name=''):
    '''
    :param tsPrice: pandas.Series
    :param period: int
    :param name: str
    :return: pandas.Series
    '''
    temp_index = tsPrice.index
    tsPrice = np.array(tsPrice)
    exponential = 2 / (period + 1)
    ema = np.ones_like(tsPrice)
    ema[0] = tsPrice[0]
    for i in range(1, len(tsPrice)):
        ema[i] = exponential * tsPrice[i] + (1 - exponential) * ema[i - 1]
    ema = pd.Series(ema, index=temp_index)
    ema.name = name + '_ema_' + str(period)
    return ema


def DIF(tsPrice,short,long):
    ema_s=EMA(tsPrice,period=short,name='')
    ema_l=EMA(tsPrice,period=long,name='')
    dif=ema_s-ema_l
    dif.name='_'.join(['DIF',str(short),str(long)])
    return dif


data=np.random.randn(500)-1
data=data.cumsum()
tsPrice=pd.Series(data,index=None)+1000

Ema_5=EMA(tsPrice,period=5)
Ema_10=EMA(tsPrice,period=10)
Ema_20=EMA(tsPrice,period=20)
Ema_120=EMA(tsPrice,period=120)

dif=DIF(tsPrice,12,26)
dea=EMA(dif,period=9)

macd=(dif-dea)*2
macd_up=macd>0
macd_down=macd<=0


ax_left=0.05
ax_between=0.05
ax_ratio=3  # 图一图二的高度比
ax_width=1-2*ax_left
ax_height_1=(1-3*ax_between)*(ax_ratio/(ax_ratio+1))
ax_height_2=(1-3*ax_between)*(1/(ax_ratio+1))
ax_bottom_1=1-1*ax_between-ax_height_1
ax_bottom_2=ax_between

ax_rect_1=[ax_left,ax_bottom_1,ax_width,ax_height_1]
ax_rect_2=[ax_left,ax_bottom_2,ax_width,ax_height_2]

fig=plt.figure(figsize=(16,9))
ax_1=fig.add_axes(ax_rect_1)
ax_2=fig.add_axes(ax_rect_2,sharex=ax_1)

df_price = pd.concat([tsPrice, Ema_20, Ema_120], axis=1, sort=False)
df_price.columns=['price','ema_short','ema_long']
df_price.plot(ax=ax_1)

ax_2.vlines(macd[macd_up].index,0,macd[macd_up],colors='c')
ax_2.vlines(macd[macd_down].index,0,macd[macd_down],colors='y')
ax_2.plot(dif,color='r',label='dif')
ax_2.plot(dea,color='b',label='dea')
ax_2.legend()
plt.show()
