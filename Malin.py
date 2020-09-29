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

#%% Skalerer data:
    
# Henter ut nullverdier i eget df og lagrer som csv
NaN_draft=df_draft.copy()

NaN_draft= NaN_draft.loc[NaN_draft['Total_Sales_day']<0]

NaN_draft.to_csv('Nullvalues_totalsales.csv')



# Fjerner nullverdier fra df_concat ved å filtrere dem vekk:
df_concat= df_concat.loc[df_concat['Total_Sales_day']>=0]

# Dobbeltsjekker at det ikke er nullverdier:
df_concat['Total_Sales_day'].isna().sum()



# Skalerer ved log:

log=np.log(np.c_[df_concat['Total_Sales_day']])
log=pd.DataFrame(log)

#Sjekker nullverdier:
log.isna().sum()

# Sjekker distribusjonen etter logtransformasjon:
log.hist()


# Standardisering:   
StdScale=StandardScaler()   

StdScale.fit(log)

std_total= StdScale.transform(log)

#Sjekker fordelingen:
plt.hist(std_total)

# Setter kolonnenne tilbake i df:
df_concat['Total_Sales_day_std'] =std_total[:,0]
del df_concat['Total_Sales_day']




# Train, test, split:

# filter data for one shop
df_one_month = df.loc[df['date_block_num']== 0, :]
df_grouped_shop = df_one_month[['shop_id', 'date', 'Total_Sales_day']].groupby(by=['shop_id', 'date']).sum().reset_index()



X_one_month = np.c_[df_grouped_shop[['shop_id', 'date']]]
y_one_month = np.c_[df_grouped_shop['Total_Sales_day']]



X_train, X_test, y_train, y_test = train_test_split(
    X_one_month, y_one_month, test_size=0.3, random_state=420)

X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.333, random_state=420)
