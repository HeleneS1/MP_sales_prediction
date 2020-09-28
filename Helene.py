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

df_y = df_original[['date', 'shop_id']]
df_X = df_original[['date', 'item_id', 'item_id', 'item_cnt_day']]