""" This is the file where we connect all the dots """
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import OneHotEncoder
import datetime as dt
#%%
os.chdir('C:/Users/Helene Stabell/Desktop/Academy/Uke 9MP/')
df_original = pd.read_csv('sales_train.csv')

#Lager kopi av datasettet
df_draft = df_original.copy()
df_draft = df_draft[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']]

# Legger til ny kolonne:
df_draft['Total_Sales_day'] = df_draft['item_price'] * df_draft['item_cnt_day']

# Sjekker fordelingen:
# df_draft.hist()

#Sjekker for null-verdier:
df_draft.isna().sum()

#%%
# filter data for one shop
df_one_month = df_draft.loc[df_draft['date_block_num']== 0, :]
df_grouped_shop = df_one_month[['shop_id', 'date', 'Total_Sales_day']].groupby(by=['shop_id', 'date']).sum().reset_index()

#%%
# One Hot Encoder av shop-id:
onehot= OneHotEncoder()
store_id= df_grouped_shop[['shop_id']]
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


#%%



X_one_month = np.c_[df_concat[['shop_id', 'year','month', 'day','quarter','dayofweek', 'is_weekend' ]]]
y_one_month = np.c_[df_concat['Total_Sales_day']]


# Train, test, split:
X_train, X_test, y_train, y_test = train_test_split(
    X_one_month, y_one_month, test_size=0.3, random_state=420)

X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.333, random_state=420)


input_layer = Input(shape=(2,))
first_hidden_layer = Dense(3, activation = "relu")(input_layer)
second_hidden_layer = Dense(3, activation = "relu")(first_hidden_layer)
third_hidden_layer = Dense(2, activation = "relu")(second_hidden_layer)
output_layer = Dense(1, activation = "linear")(third_hidden_layer)

linear_model1 = Model(inputs = input_layer, outputs = output_layer)
linear_model1.compile(optimizer='adam', loss='mse', metrics=['mae'])

linear_model1.fit(X_one_month,y_one_month, batch_size=32, epochs=10)

# prediction = linear_model1.predict(X_one_month)


print(prediction)

#Om vi ønsker å sette inn prediction:
X_new = [[]] 
print(model.predict(X_new))


print(model.score(X = X, y = y))
y_pred = model.predict(X) 
print(np.sqrt(mean_squared_error(y_pred, y)))
print(mean_absolute_error(y_pred, y))

y_mean = np.mean(y)*np.ones(y.shape)#mean = gjennomsnittet
print(np.sqrt(mean_squared_error(y_mean, y)))#np.sqrt = kvadratroten
print(mean_absolute_error(y_mean, y)) 


#Om man ønsker å laget et plot:
plt.plot(history.history['loss'], label='Kostnadder på treningsdataene')
plt.plot(history.history['val_loss'], label='Kostnadder på testdataene')#denne skal være så lik treningsdataene som mulig
plt.legend(loc='upper right')
plt.show()

# make each plot seperatly 
plt.plot(history.history['acc'], label='treningsdata nøyaktighet')
plt.plot(history.history['val_acc'], label='testdata nøyaktighet')
plt.legend(loc='upper right')
plt.show()

