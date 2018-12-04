# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# #seed用法
# np.random.seed(1)
# r = np.random.standard_normal((4,3))
# print(r)
# np.random.seed(2)
# s = np.random.standard_normal((4,3))
# print(s)
# def f(x):
#     return 3 * x + 5
# print(f(s))
# print(np.sin(s))
#
#
# y = np.random.standard_normal(30)
# print(y.reshape(6,5))
# #ndarray 无法直接画图，而dataframe可以,c是color,lw是linewidth
# #pd.DataFrame(y)[0].plot(grid = True,c = 'r',lw = 3)
# #相同作用
# plt.plot(y,lw = 3,c = 'r', linestyle = ':',marker = 'o',label = 'aaa',)
# plt.grid(True)
# plt.axis('tight')
# plt.xlim(-1,31)
# plt.ylim(np.min(y)-1,np.max(y)+1)
# plt.xlabel('index')
# plt.ylabel('value')
# plt.title('simple plot')
# #显示标签需要加这一句
# plt.legend()
# #plt.show()
#
# np.random.seed(3)
# y = np.random.standard_normal((20,2)).cumsum(axis=0)
# print(y)
# #plot two lines
# plt.figure(figsize=(7,4))
# plt.plot(y[:,0],color = 'b', lw = 1.5,linestyle = '-',label = '1st')
# plt.plot(y[:,1],color = 'y', linestyle = '--',label = '2nd')
# plt.grid(True)
# plt.axis('tight')
# #loc为位置，右上-左上-左下
# plt.legend(loc = 3)
# #plt.show()
#
#
# #如果两个y数量级不同，引入第二条y轴
# #fig,ax 访问底层绘图对象
# fig,ax1 = plt.subplots()
# print(fig,ax1)
# #一列数据放大100倍
# y[:,1 ]= y[:,1 ]*100
# plt.plot(y[:,0],color = 'b', lw = 1.5,linestyle = '-',label = '1st')
# plt.grid(True)
# plt.axis('tight')
# plt.legend(loc = 8)
# plt.ylabel('1st value')
# ax2 = ax1.twinx()
# plt.plot(y[:,1],color = 'y', lw = 2,linestyle = '--',label = '2nd')
# plt.ylabel('2nd value')
# plt.legend(loc = 3)
# #plt.show()
#
#
# #单独两个子图
# plt.figure(figsize=(8,5))
# plt.subplot(211)
# plt.plot(y[:,0],color = 'b', lw = 1.5,linestyle = '-',label = '1st')
# plt.grid(True)
# plt.legend(loc = 0)
# plt.axis('tight')
# plt.ylabel('value')
# plt.title('simple')
#
# plt.subplot(212)
# plt.plot(y[:,1],color = 'y', lw = 2,linestyle = '--',label = '2nd')
# plt.legend(loc = 0)
# plt.axis('tight')
# plt.ylabel('value')
# plt.xlabel('index')
# plt.ylabel('value')
# # plt.show()
#
#
#
# plt.figure(figsize=(9,4))
# plt.subplot(121)
# plt.plot(y[:,0],color = 'b', lw = 1.5,linestyle = '-',label = '1st')
# plt.grid(True)
# plt.legend(loc = 0)
# plt.axis('tight')
# plt.ylabel('value')
# plt.title('1st')
# plt.subplot(122)
# plt.bar(np.arange(len(y)),y[:,1],color = 'y', lw = 2,linestyle = '--',label = '2nd')
# plt.grid(True)
# plt.legend(loc = 0)
# plt.axis('tight')
# plt.ylabel('value')
# plt.title('2nd')
# # plt.show()
#
#
# z = np.random.standard_normal((1000,2))
# plt.figure(figsize=(8,5))
# plt.plot(z[:,0],z[:,1],'ro')
# plt.grid(True)
# plt.xlabel('1st')
# plt.ylabel('2nd')
# plt.title('scatter')
#
# #bins为数据组
# plt.figure(figsize=(8,5))
# plt.hist(z,label=['1st','2nd'],bins = 25)
# plt.grid(True)
# plt.xlabel('value')
# plt.ylabel('frequency')
# plt.title('hist')
# # plt.show()
#
#
# #箱型图
# fig,ax = plt.subplots(figsize = (8,5))
# plt.boxplot(z)
# plt.grid(True)
# plt.setp(ax,xticklabels = ['1st','2nd'])
# plt.xlabel('dataset')
# plt.ylabel('value')
# plt.title('box')
# plt.show()
# #
# import numpy as np
# from matplotlib.patches import  Polygon
# import matplotlib.pyplot as plt
# def func(x):
#     return 0.5*np.exp(x)+1
# a,b = 0.5,1.5
# x = np.linspace(0,2)
# y = func(x)
# fig,ax = plt.subplots(figsize = (8,5))
# plt.plot(x , y ,'b',linewidth = 2)
# plt.ylim(ymin = 0)
# Ix = np.linspace(a,b)
# Iy = func(Ix)
# verts = [(a,0)]+list(zip(Ix,Iy))+[(b,0)]
# poly = Polygon(verts,facecolor='0.7',edgecolor='0.5')
# ax.add_patch(poly)
# plt.text(0.5*(a+b),1,'$\int_a^b f(x)\mathrm{d}x$',horizontalalignment = 'center',fontsize = 20)
# plt.figtext(0.9,0.075,'$x$')
# plt.figtext(0.075,0.9,'$f(x)$')
# ax.set_xticks((a,b))
# ax.set_xticklabels(('$a$','$b$'))
# ax.set_yticks([func(a),func(b)])
# ax.set_yticklabels(('$f(a)$','$f(b)$'))
# plt.grid(True)
# plt.show()




'''
#k线图
#https://blog.csdn.net/u014281392/article/details/73611624
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from matplotlib.pylab import date2num
import matplotlib.pyplot as plt
import datetime
quotes = web.DataReader('^GSPC', data_source='yahoo',
                       start='5/1/2014', end='6/30/2014').reset_index()
quotes['date'] = quotes['Date'].apply(lambda x:date2num(datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S')))
quotes = quotes[['date','Open','Close','High','Low','Volume']]
#一定要将dataframe转换为二维数组才能带入quotes中
newdata = quotes.as_matrix()
import mpl_finance as mpf
fig,ax = plt.subplots(figsize = (8,5))
mpf.candlestick_ochl(ax,newdata,width = 0.6,colorup='r',colordown='g',alpha=1.0)
plt.grid(True)
# myFmt = mdates.DateFormatter('%Y%m%d')
# ax.xaxis.set_major_formatter(myFmt)
ax.xaxis_date()
# ax.autoscale_view()
# plt.gcf().autofmt_xdate()
plt.ylabel('Price')
plt.xlabel('Date')
plt.setp(plt.gca().get_xticklabels(),rotation = 30)
plt.show()
'''

#3d图
import numpy as np
import matplotlib.pyplot as plt
# strike = np.linspace(50,150,24)
# ttm = np.linspace(0.5,2.5,24)
# strike,ttm = np.meshgrid(strike,ttm)
# iv = (strike - 100)**2/(100*strike)/ttm
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(8,5))
# ax = fig.gca(projection = '3d')
# surf = ax.plot_surface(strike,ttm,iv,rstride=2,cstride=2,cmap=plt.cm.coolwarm,linewidth=0.5,antialiased = True)
# ax.set_xlabel('strike')
# ax.set_ylabel('time')
# ax.set_zlabel('vola')
# fig.colorbar(surf,shrink = 0.5,aspect = 5)
# plt.show()

#移动平均等
# import pandas as pd
# import pandas_datareader.data as web
# DAX = web.DataReader(name = '^GDAXI',data_source='yahoo',start = '2000-1-1',end = '2014-9-26')
# DAX['Close'].plot(figsize = (8,5))
# DAX['Return'] = np.log(DAX['Close']/DAX['Close'].shift(1))
# DAX[['Close','Return']].plot(subplots = True,style = 'b',figsize = (8,5))
# DAX['42d'] = DAX['Close'].rolling(window = 42).mean()
# DAX['42dmax'] = DAX['Close'].rolling(window = 42).max()
# DAX['252d'] = DAX['Close'].rolling(window = 252).mean()
# #print(DAX.tail(20))
# DAX[['Close','42d','42dmax','252d']].plot(figsize = (8,5))
# import math
# DAX['MOV_VOL'] = DAX['Return'] .rolling(window = 252).std()*math.sqrt(252)
# DAX[['Close','Return','MOV_VOL']].plot(subplots = True,style = 'r',figsize = (8,5))
# plt.show()

#回归分析
import pandas as pd
import urllib.request

es_url = 'https://www.stoxx.com/document/Indices/Current/HistoricalData/hbrbcpe.txt'
vs_url = 'https://www.stoxx.com/document/Indices/Current/HistoricalData/h_vstoxx.txt'

es_txt = 'es.txt'
vs_txt = 'vs.txt'
urllib.request.urlretrieve(es_url, es_txt)
urllib.request.urlretrieve(vs_url, vs_txt)
lines = open(es_txt, 'r').readlines()
lines = [line.replace(' ', '') for line in lines]
new_file = open('es50.txt','w')
new_file.writelines('date' + lines[3][:-1] + ';DEL' + lines[3][-1])
new_file.writelines(lines[4:])
new_file.close()
new_lines = open('es50.txt','r').readlines()
es = pd.read_csv('es50.txt',index_col= 0 ,parse_dates=True , sep=';', dayfirst=True)
cols = ['SX5P', 'SX5E', 'SXXP', 'SXXE', 'SXXF', 'SXXA', 'DK5F', 'DKXF']
es2 = pd.read_csv(es_url,index_col=0,parse_dates=True,sep=';',dayfirst=True,header=None,skiprows=4,names=cols)

vs = pd.read_csv('vs.txt',index_col=0,header=2,parse_dates=True,sep=',',dayfirst=True)

import datetime as dt
data = pd.DataFrame({'EUROSTOXX':es2['SX5E'][es2.index > dt.datetime(1999,1,1)]})

data = data.join(pd.DataFrame({'VSTOXX':vs['V2TX'][vs.index > dt.datetime(1999,1,1)]}))
print(data.index)
data = data.fillna(method='ffill')#ffill向前填充，bfill向后填充
print (data)
data.plot(subplots = True,grid = True,style = 'g',figsize = (8,5))
plt.show()