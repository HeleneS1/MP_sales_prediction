import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


data= pd.read_csv('sales_train.csv')

#Lager kopi av datasettet
df_draft= data.copy()


# Legger til ny kolonne:
df_draft['Total_Sales_day']=df_draft['item_price']* df_draft['item_cnt_day']


# Sjekker fordelingen:
df_draft.hist()


#Sjekker for null-verdier:
df_draft.isna().sum()



# One Hot Encoder av shop-id:
onehot= OneHotEncoder()

store_id= df_draft[['shop_id']]

onehot.fit(store_id)
np_onehot=onehot.transform(store_id).todense()

# Antar at kategoriene er "shopnr"
print(onehot.categories_)

#OHE som eget dataframe
ohe_shop= pd.DataFrame(np_onehot,columns=onehot.categories_)

# Legger sammen med OG-dataframe
df_concat= pd.concat([df_draft, ohe_shop],axis=1)


#%%
# Fikser dato
# OHE Dato
import datetime as dt

# Gjør om date til datetime-format:
df_concat['date']= pd.to_datetime(df_concat['date'],format= '%d.%m.%Y')


# Legger til dag, måned, år, kvartal:
df_concat['month'] = df_concat['date'].dt.month

df_concat['year'] = df_concat['date'].dt.year

df_concat['day'] = df_concat['date'].dt.day

df_concat['quarter'] = df_concat['date'].dt.quarter


# Legger til day of week(numerisk) og tekst. 
#NB! Mandag=0, søndag=6
df_concat['dayofweek'] = df_concat['date'].dt.dayofweek # Numerisk

df_concat['dayofweek_text'] = df_concat['date'].dt.day_name() #Gir i tekst


# Sjekker om det er helg eller ikke:
# np.where()= hvis sant, gjør x, ellers y
df_concat['is_weekend']= np.where(df_concat['dayofweek_text'].isin(['Sunday','Saturday']),1,0)

