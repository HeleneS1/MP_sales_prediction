import pandas as pd
import numpy as np
import os 

#%% Import data 
os.chdir('C:/Users/Helene Stabell/Desktop/Academy/Uke 9MP/')

df_original = pd.read_csv('sales_train.csv')


#%%
# filter data based on day
df_original.columns
#['date', 'date_block_num', 'shop_id', 'item_id', 'item_price','item_cnt_day']


#%%
# filter data for one shop

df = df_original[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']]
df['Total_Sales_day'] = df['item_price'] * df['item_cnt_day']

df_one_month = df.loc[df['date_block_num']== 0, :]
df_grouped_shop = df_one_month[['shop_id', 'date', 'Total_Sales_day']].groupby(by=['shop_id', 'date']).sum().reset_index()

#%%

X_one_month = np.c_[df_grouped_shop[['shop_id', 'date']]]
y_one_month = np.c_[df_grouped_shop['Total_Sales_day']]


