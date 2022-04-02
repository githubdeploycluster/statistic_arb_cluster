#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('mkdir tmp')


# In[ ]:


get_ipython().system('ls -la')


# In[1]:


class SignalModel(object):

  def __init__(self) -> None:
      pass
    
  def crossover(self, x, y):
    res = [False]
    for i in range(1, len(x)):
      if x[i] > y[i] and x[i-1] < y[i-1]:
        res.append(True)
      else:
        res.append(False)
    return res


  def crossunder(self, x, y):
    res = [False]
    for i in range(1, len(x)):
      if x[i] < y[i] and x[i-1] > y[i-1]:
        res.append(True)
      else:
        res.append(False)
    return res


  def get_orderbook(self, df, ticker_1, ticker_2, comission=0.003):
      order_list = list()
      order_list_1 = list()

      order_1 = dict()
      order_2 = dict()
      order_tmp = dict()
      deal_id = 0

      position = None
      for index, row in df.iterrows():

          if row['signal'] == "Open Long" and not position:
              deal_id += 1
              order_1['buy'] = row[ticker_1]
              order_1['datetime_buy'] = index
              order_1['deal_id'] = deal_id
              order_1['ticker'] = ticker_1.split('_')[0]
              order_1['position_open'] = 'open long'

              order_tmp['ticker'] = ticker_1.split('_')[0]
              order_tmp['price'] = row[ticker_1]
              order_tmp['datetime'] = index
              order_tmp['position_open'] = 'open long'
              order_list_1.append(order_tmp)
              order_tmp = dict()
              #             print("OPEN LONG")

              order_2['shell'] = row[ticker_2]
              order_2['datetime_shell'] = index
              order_2['deal_id'] = deal_id
              order_2['ticker'] = ticker_2.split('_')[0]
              order_2['position_open'] = 'open short'

              order_tmp['ticker'] = ticker_2.split('_')[0]
              order_tmp['price'] = row[ticker_2]
              order_tmp['datetime'] = index
              order_tmp['position_open'] = 'open short'
              order_list_1.append(order_tmp)
              order_tmp = dict()

              position = True

          if row['signal'] == "Close Long" and position:
              order_1['shell'] = row[ticker_1]
              order_1['datetime_shell'] = index
              order_1['ticker'] = ticker_1.split('_')[0]
              order_1['position_close'] = 'close long'
              order_list.append(order_1)

              order_tmp['ticker'] = ticker_1.split('_')[0]
              order_tmp['price'] = row[ticker_1]
              order_tmp['datetime'] = index
              order_tmp['position_open'] = 'close long'
              order_list_1.append(order_tmp)
              order_tmp = dict()

              order_2['buy'] = row[ticker_2]
              order_2['datetime_buy'] = index
              order_2['ticker'] = ticker_2.split('_')[0]
              order_2['position_close'] = 'close short'
              order_list.append(order_2)

              order_tmp['ticker'] = ticker_2.split('_')[0]
              order_tmp['price'] = row[ticker_2]
              order_tmp['datetime'] = index
              order_tmp['position_open'] = 'close short'
              order_list_1.append(order_tmp)
              order_tmp = dict()

              order_1 = dict()
              order_2 = dict()

              position = None

          if row['signal'] == "Open Short" and not position:
              deal_id += 1
              order_2['buy'] = row[ticker_2]
              order_2['datetime_buy'] = index
              order_2['deal_id'] = deal_id
              order_2['ticker'] = ticker_2.split('_')[0]
              order_2['position_open'] = 'open long'

              order_tmp['ticker'] = ticker_2.split('_')[0]
              order_tmp['price'] = row[ticker_2]
              order_tmp['datetime'] = index
              order_tmp['position_open'] = 'open long'
              order_list_1.append(order_tmp)
              order_tmp = dict()

              order_1['shell'] = row[ticker_1]
              order_1['datetime_shell'] = index
              order_1['deal_id'] = deal_id
              order_1['ticker'] = ticker_1.split('_')[0]
              order_1['position_open'] = 'open short'

              order_tmp['ticker'] = ticker_1.split('_')[0]
              order_tmp['price'] = row[ticker_1]
              order_tmp['datetime'] = index
              order_tmp['position_open'] = 'open short'
              order_list_1.append(order_tmp)
              order_tmp = dict()

              position = True

          if row['signal'] == "Close Short" and position:
              order_2['shell'] = row[ticker_2]
              order_2['datetime_shell'] = index
              order_2['ticker'] = ticker_2.split('_')[0]
              order_2['position_close'] = 'close long'
              order_list.append(order_2)

              order_tmp['ticker'] = ticker_2.split('_')[0]
              order_tmp['price'] = row[ticker_2]
              order_tmp['datetime'] = index
              order_tmp['position_open'] = 'close long'
              order_list_1.append(order_tmp)
              order_tmp = dict()

              order_1['buy'] = row[ticker_1]
              order_1['datetime_buy'] = index
              order_1['ticker'] = ticker_1.split('_')[0]
              order_1['position_close'] = 'close short'
              order_list.append(order_1)

              order_tmp['ticker'] = ticker_1.split('_')[0]
              order_tmp['price'] = row[ticker_1]
              order_tmp['datetime'] = index
              order_tmp['position_open'] = 'close short'
              order_list_1.append(order_tmp)
              order_tmp = dict()

              order_1 = dict()
              order_2 = dict()

              position = None

      order_df = pd.DataFrame(order_list)

      if len(order_df):
          order_df['PP'] = (order_df['shell'] - order_df['buy'] -                             order_df['shell'] * comission - order_df['buy'] * comission) / order_df['buy']

      order_df_1 = pd.DataFrame(order_list_1)

      return order_df, order_df_1

  def get_signal(self, df, value, col):
      signal = df[col].tolist()

      long_condition_crossunder = [-value for i in range(len(signal))]
      short_condition_crossunder = [value for i in range(len(signal))]

      longCondition = self.crossunder(signal, long_condition_crossunder)
      shortCondition = self.crossover(signal, short_condition_crossunder)

      res = list()
      position = ""

      for i in range(len(df)):
          _res = None
          if position == "" and longCondition[i]:
              position = "Long"
              _res = "Open Long"
          if position == "" and shortCondition[i]:
              position = "Short"
              _res = "Open Short"

          if signal[i] > 0 and position == "Long":
  #                 print(f">>>>>>>")
              position = ""
              _res = "Close Long"
          if signal[i] < 0 and position == "Short":
              position = ""
              _res = "Close Short"
          res.append(_res)

      return res


# In[2]:


import pandas as pd
import numpy as np
import os
import json
import datetime
import time
import threading
import statsmodels.api as sm
import requests
from sklearn.preprocessing import StandardScaler

# import signal_model as signal_model
signal_model = SignalModel()


# import openapi
# from openapi_client import openapi

from sklearn import preprocessing
import math

from tqdm import tqdm
from importlib.machinery import SourceFileLoader


import warnings
warnings.filterwarnings("ignore")


# secret_file = '../../../secret/path.json'
# with open(secret_file, 'r') as file:
#     cfg = json.load(file)
    
# with open('../env.txt', 'r') as file:
#     env = file.read()

# base_producer = SourceFileLoader('base_producer', f"{cfg['libs_dev_path']}/base_producer.py").load_module()
# import base_producer as base_producer

# parser_finam_date = SourceFileLoader('parser_finam_date', f"{cfg['code_path']}/{env}/parser_finam_date/parser_finam_date.py").load_module()
# import parser_finam_date as parser_finam_date

# bablofil_ta = SourceFileLoader('bablofil_ta', f"{cfg['libs_dev_path']}/bablofil_ta.py").load_module()
# import bablofil_ta as ta

# parser_stock = SourceFileLoader('parser_stock', f"{cfg['code_path']}/dev/parser_stock/parser_stock.py").load_module()
# import parser_stock


# In[3]:


class ClusterServiseClient:

    def __init__(self, host='95.165.139.159', port='5001'):
        self.host = f"http://{host}"
        self.port = port

    def check_done_ticker(self, ticker_1, ticker_2, currency):
        data = {'ticker_1': ticker_1, 'ticker_2': ticker_2, 'currency': currency}
        response = requests.get(f"{self.host}:{self.port}/check_done_ticker", data=data)
        return response.text

    def set_result(self, ticker_1, ticker_2, path, file_name):
        # ticker_1 = params['ticker_1']
        # ticker = params['ticker']
        # day_learn = params['day_learn']
        # test_interval = params['test_interval']

        filename = f"{ticker_1}-{ticker_2}.csv"
        data = {'ticker_1': ticker_1, 'ticker_2': ticker_2, 'currency': currency}
        data['filename'] = filename
        

        file = {file_name: open(f"{path}/{file_name}", 'rb')}

        responce = requests.post(f"{self.host}:{self.port}/set_result", data=data, files=file)
#         if responce.status_code == 200:
#             print(file_name)
            # print(f">>> {res.status_code}")


# In[4]:


# !mkdir tmp


# In[5]:


timeframe = '10min'
type_ = 'Stock'
provider = 'tinkoff'

# comission = 0.003
comission = 0.0004

corr_k = 0.0

window = 200

currency = "RUB"

date_start = "2021-06-01 00:00:00"
date_end = "2022-01-01 00:00:00"

date_start_test = "2022-01-01 00:00:00"
date_end_test = "2022-06-01 00:00:00"

path_to_orders_file = f'csv/all_calc/{currency}_order'


# In[6]:


# parser_finam_date.parser_mode(timeframe=timeframe)
# parser_finam_date.update_mode(timeframe=timeframe)


# In[7]:


# datetime_list = parser_finam_date.get_date_list(timeframe=timeframe)


# In[8]:


# # Получение дат с finam
# date_df = pd.DataFrame()
# date_df['datetime'] = parser_finam_date.get_date_list(timeframe=timeframe)


# ## Получение списка тикеров из базы

# In[9]:


host = 'http://95.165.139.159'
base_port = '5002'
data = {'currency': currency, 'type': 'Stock'}
data = requests.get(f"{host}:{base_port}/get_all_ticker", data=data)
ticker_list = json.loads(data.text)
print(f"Всего активов: {len(ticker_list)}")


# ## Загрузка котировок в оперативную память

# In[ ]:


def get_all_data_ticekr(ticker_list, timeframe):
    all_ticker_dict = dict()
    
    for ticker in tqdm(ticker_list):
        try:
            data={
                'provider': 'tinkoff',
                'type': 'Stock',
                'ticker': ticker,
                'timeframe': timeframe,
                'crop_options': 'False'
            }
            url = f"{host}:{base_port}/get_all_data_ticekr"
            res = json.loads(requests.post(url=url, data=data).text)
            df = pd.DataFrame(res)
            df = df.set_index('datetime')

        except:
            df = pd.DataFrame()
        if len(data):
            all_ticker_dict[ticker] = df
        
    return all_ticker_dict


# In[ ]:


def fast_get_data(all_data_ticker, ticker, date_start, date_end, price_filter=50):
    if all_data_ticker[ticker]['close'].tolist()[0] < price_filter:
        data = all_data_ticker[ticker].copy()
    #     print(data.head())
        data = data.loc[data['datetime'] >= date_start]
        data = data.loc[data['datetime'] <= date_end]
        return data
    else:
        return pd.DataFrame()


# In[ ]:





# In[ ]:


all_data_ticker = get_all_data_ticekr(
    ticker_list = ticker_list,
    timeframe = timeframe
)


# In[ ]:


# all_data_ticker['SBER'].head()


# In[ ]:





# ## Комбинаторика

# In[ ]:


import itertools

perm_set = itertools.permutations(all_data_ticker.keys(), 2)
res_ticker_list = list()
stop_ticker_list = list()

_len = 0
for i in tqdm(perm_set): _len += 1
    
print(f"len perm_set: {_len}")

perm_set = itertools.permutations(all_data_ticker.keys(), 2)
for i in tqdm(perm_set):
#     if stop_ticker_list:
    res_ticker_list.append((i[0], i[1]))

#     stop_ticker_list.append(i[0])
#     stop_ticker_list = list(set(stop_ticker_list))
    
print(f"res len: {len(res_ticker_list)}")


# In[ ]:


all_pair_stock = list()
for item in tqdm(res_ticker_list):
    all_pair_stock.append((item[0], item[1]))


# In[ ]:


all_pair_stock = pd.DataFrame(all_pair_stock)
all_pair_stock = all_pair_stock.rename(columns={0: 'ticker_1', 1: 'ticker_2', 2: 'corr'})
all_pair_stock['value']  = [0.2 for i in range(len(all_pair_stock))]


# ## Функции

# In[ ]:


def get_final_list_test(stock_df, ticker_1, ticker_2):
    final_list_test = list()
    orderbook_df_dict_test = dict()
    orderbook_df_1_test = pd.DataFrame()
    orderbook_df_test = pd.DataFrame()

#     for index, row in all_pair_stock.iterrows():
#     ticker_1 = row['ticker_1']
#     ticker_2 = row['ticker_2']
#     value = row['value']
    value = 0.02


    filter_df = filter_df_on_nan(df=stock_df, nan_percent=1)
#     print(filter_df.head())
    try:
        norm_df = normalization(df=filter_df)
    except:
        return orderbook_df_test, orderbook_df_1_test, final_list_test


    #     print(ticker_1, ticker_2)
    #     try:
    if len(norm_df.columns) > 1:
        res_df_test, orderbook_df_test, orderbook_df_1_test_ = check_result(
            ticker_1=ticker_1,
            ticker_2=ticker_2,
            value=value,
            norm_df=norm_df,
            df=stock_df
        )

        orderbook_df_1_test = pd.concat([orderbook_df_1_test, orderbook_df_1_test_], ignore_index=True)

        #     print(orderbook_df.head())

        #     print(res_df_test)
        if len(res_df_test):
            final_list_test.append({
                'ticker_1': ticker_1,
                'ticker_2': ticker_2,
                'value': res_df_test[0]['value'],
                'pp': res_df_test[0]['PP'],
                'len_deals': res_df_test[0]['len_deals'],
                'pp_final': res_df_test[0]['pp_final'],

            })

    return orderbook_df_test, orderbook_df_1_test, final_list_test


def filter_df_on_nan(df, nan_percent=0.2):
    df_tmp = df.copy()
#     print(f"nan_percent: {nan_percent}")
    column_list = list()
    
    for col_name in df_tmp.columns:
        nan = get_nan_lenth(col=df_tmp[col_name].tolist())
        if nan/len(df) < nan_percent:
            column_list.append(col_name)
            df_tmp[col_name] = replace_na_on_mean_value(col=df_tmp[col_name].tolist())
    
    res = df_tmp[column_list]
    return res


def get_nan_lenth(col):
    index = 0
    
    for i in col:
        if math.isnan(i):
            index += 1
            
    return index


def replace_na_on_mean_value(col):
    index_list = list()
    first_value = None
    last_value = None
    
    for i in range(len(col)):
        
        if math.isnan(col[i]) and first_value == None:
            index_list.append(i)
            
        elif not math.isnan(col[i]) and len(index_list) and first_value == None:
            first_value = col[i]
            for index in index_list:
                col[index] = col[i]
            index_list = list()
            
            
        elif math.isnan(col[i]) and first_value != None:
            index_list.append(i)
        elif not math.isnan(col[i]) and first_value != None and len(index_list):
            mean = (first_value + col[i]) / 2
            for index in index_list:
                col[index] = mean
                
            index_list = list()
            first_value = col[i]
            last_value = None
            
#         else:
#             print(f">>>>>>")
            
    
#     print(col[:5])
    return col

def normalization(df):
    norm_df = df.copy()
    for col_name in norm_df.columns:
        x_array = np.array(norm_df[col_name])
        normalized_arr = preprocessing.normalize([x_array])
        norm_df[col_name] = normalized_arr[0]
#         print(normalized_arr[0])

    return norm_df


def check_result(ticker_1, ticker_2, value, norm_df, df):
    res_list = list()

    # Построение сигналов
    signal_df = pd.DataFrame()
    signal_df['signal'] = norm_df[ticker_1] / norm_df[ticker_2] - 1
    #
    # print(signal_df)
    signal = signal_model.get_signal(df=signal_df, value=value, col='signal')

    #         df['signal'] = signal_model(df=df, ticker_1=ticker_1, ticker_2=ticker_2, value=value)
    calc_df = norm_df
    calc_df['signal'] = signal
    
    calc_df = calc_df.merge(df, how='right', left_index=True, right_index=True)
    
    
    _ = list()
    for index, row in calc_df.iterrows():
        if row['signal'] != None:
            _.append(row)
    _ = pd.DataFrame(_).dropna()

    # Рассчет сделок
    orderbook_df, orderbook_df_1 = signal_model.get_orderbook(df=_, ticker_1=f"{ticker_1}_y", ticker_2=f"{ticker_2}_y")

#     print(orderbook_df_1.head())

    PP = 0
    len_deals = 0

    if len(orderbook_df):
        PP = orderbook_df['PP'].sum()
        len_deals = len(orderbook_df.groupby(['deal_id']).sum())

        orderbook_df['datetime_shell'] = pd.to_datetime(orderbook_df['datetime_shell'])
        orderbook_df['datetime_buy'] = pd.to_datetime(orderbook_df['datetime_buy'])
        orderbook_df['time_in_deal'] = orderbook_df['datetime_shell'] - orderbook_df['datetime_buy']

        delta_time = datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

        pp_final_list = list()

        for index, row in orderbook_df.iterrows():
            if row['time_in_deal'] < delta_time:
                pp_final = row['PP'] - 0.090 / 100 * (abs(row['time_in_deal'].days) + 1)
            else:
                pp_final = row['PP']

            pp_final_list.append(pp_final)

        # orderbook_df['time_in_deal_mode'] = time_in_deal_list
        orderbook_df['PP_final'] = pp_final_list
        pp_final = orderbook_df['PP_final'].sum()

        res_list.append({'value': value, 'PP': PP, 'len_deals': len_deals, 'pp_final': pp_final})

    return res_list, orderbook_df, orderbook_df_1

def check_ticker_position(ticker_position, old_ticker_position):
    new = dict()
    for ticker_old in old_ticker_position.keys():
        if ticker_position[ticker_old]['position'] != old_ticker_position[ticker_old]['position']:
            new[ticker_old] = ticker_position[ticker_old]
#             print(ticker_position[ticker_old])
        
#     value = { k : ticker_position[k] for k in set(ticker_position) - set(old_ticker_position) }
    return new


# In[ ]:





# # Рассчет модели

# In[ ]:


def calc_tow_stock(df, ticker_1, ticker_2):
    index = 0
#     print(f"START: {ticker_1} - {ticker_2}")
    ticker_position = dict()
    old_ticker_position = dict()
    
    orders_list = list()
    df_1 = pd.DataFrame()
    df_2 = pd.DataFrame()
    if len(df[ticker_1]) > 0 and len(df[ticker_2]) > 0:
        df_1[ticker_1]=df[ticker_1]['open']
        df_2[ticker_2]=df[ticker_2]['open']
    else:
        path = 'tmp'
        file_name = f'{ticker_1}-{ticker_2}.csv'
        
        pd.DataFrame().to_csv(f"{path}/{file_name}", sep=";")
        cluster_client.set_result(ticker_1=ticker_1, ticker_2=ticker_2, path=path, file_name=file_name)
        
    
    data = pd.merge(df_1, df_2, how='outer', left_index=True, right_index=True)
    
#     print(f">>> len : {len(data)}")
    index = 0
    for i in range(len(data)):
        stock_df = data[i:i+window]
#         if index >2:
#             break
#         else:
#             index += 1
        
        
#         try:
        orderbook_df, orderbook_df_1_test, final_list_test = get_final_list_test(
            stock_df=stock_df, 
            ticker_1=ticker_1,
            ticker_2=ticker_2
        )
#         except:
#             continue
#         print("______")
#         print(f"{len(tmp_df.columns)} | {len(filter_df.columns)}")
#         print(tmp_df.tail())
#         print(tmp_df['LFC'].tail())
#         print("______")
        
        
        if len(orderbook_df_1_test):
            orderbook_df_1_test = orderbook_df_1_test.sort_values("datetime", ascending=True)
    #         print("_________________")
    #         print(f"last datetime: {filter_df.index.values[-1]} | len: {len(orderbook_df_1_test)}")
    #         print(orderbook_df_1_test.tail())


            _ticekr_list = orderbook_df_1_test['ticker'].drop_duplicates().tolist()
    #         print(_ticekr_list)
            for ticker in _ticekr_list:
#                 print(f">>>> {ticker}")
    
#                 print(orderbook_df_1_test.loc[orderbook_df_1_test['ticker'] == ticker]['position_open'].tolist()[-1])
#                 print(orderbook_df_1_test.loc[orderbook_df_1_test['ticker'] == ticker].tail())
        
                position = orderbook_df_1_test.loc[orderbook_df_1_test['ticker'] == ticker]['position_open'].tolist()[-1]
                _datetime = orderbook_df_1_test.loc[orderbook_df_1_test['ticker'] == ticker]['datetime'].tolist()[-1]
                price = orderbook_df_1_test.loc[orderbook_df_1_test['ticker'] == ticker]['price'].tolist()[-1]
    #             break
                
        
                if stock_df.index.tolist()[-1] == _datetime:
                    ticker_position[ticker] = {'position': position, 'datetime': _datetime, 'price': price}

            new = check_ticker_position(ticker_position, old_ticker_position)
            old_ticker_position = ticker_position.copy()

            if len(new): 
                for ticker in new.keys():
                    
                    if stock_df.index.tolist()[-1] == new[ticker]['datetime']:
#                         print(f"{stock_df.index.tolist()[-1]} - {new[ticker]['datetime']}")
#                         print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   *** OK ***")
                        orders_list.append({
                            'ticker': ticker,
                            'datetime': new[ticker]['datetime'],
                            'price': new[ticker]['price'],
                            'position': new[ticker]['position'],
                        })
#                         break
#                     else:
#                         pass
#                         print(f"{stock_df.index.tolist()[-1]} - {new[ticker]['datetime']}")

#                         print(f">>>> {ticker}")
#                         print(orderbook_df_1_test.loc[orderbook_df_1_test['ticker'] == ticker].tail())   
#                         print("***********")
#                         print(stock_df.tail())

#                         print(json.dumps(new, indent=4, sort_keys=True))
#                         print("_________")
        
        
        

        index += 1
#         if index == 100:
#             break
            
#     print(orders_list)
    if len(orders_list) > 0:
#         print(f"ticker_1: {ticker_1, ticker_2}")
        path = 'tmp'
        file_name = f'{ticker_1}-{ticker_2}.csv'
        
        pd.DataFrame(orders_list).to_csv(f"{path}/{file_name}", sep=";")
        cluster_client.set_result(ticker_1=ticker_1, ticker_2=ticker_2, path=path, file_name=file_name)
#         print(f"{ticker_1} - {ticker_2}")
    else:
        path = 'tmp'
        file_name = f'{ticker_1}-{ticker_2}.csv'
        
        pd.DataFrame().to_csv(f"{path}/{file_name}", sep=";")
        cluster_client.set_result(ticker_1=ticker_1, ticker_2=ticker_2, path=path, file_name=file_name)
        
        
    return pd.DataFrame(orders_list)





# In[ ]:


# order = calc_tow_stock(
#     df=all_data_ticker,
#     ticker_1="BILI",
#     ticker_2="GTHX"
# )


# In[ ]:


# order


# In[ ]:





# In[ ]:


# # TAG

orders = dict()

cluster_client = ClusterServiseClient()

print(f"Всего итераций: {len(all_pair_stock)}")
for index, row in tqdm(all_pair_stock.iterrows()):
    if '@' not in row['ticker_1'] and '@' not in row['ticker_2']:
        if cluster_client.check_done_ticker(
            ticker_1=row['ticker_1'], 
            ticker_2=row['ticker_2'], 
        currency=currency) == 'True':
            
            order = calc_tow_stock(
                df=all_data_ticker,
                ticker_1=row['ticker_1'],
                ticker_2=row['ticker_2']
            )
    else:
        print(f"Error; {row['ticker_2']}-{row['ticker_2']}")
#         orders[f"{row['ticker_1']}-{row['ticker_2']}"] = order
        


# In[ ]:


# cluster_client = ClusterServiseClient()

# ticker_1 = 'ABRD'
# ticker_2 = 'AFKS'
# path = 'tmp'
# file_name = file_name = f'{ticker_1}-{ticker_2}.csv'

# cluster_client.set_result(ticker_1=ticker_1, ticker_2=ticker_2, path=path, file_name=file_name)

