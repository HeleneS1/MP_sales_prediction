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