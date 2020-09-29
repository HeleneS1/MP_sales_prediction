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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import datetime as dt
from sklearn.model_selection import train_test_split
#%%
os.chdir('C:/Users/Helene Stabell/Desktop/Academy/Uke 9MP/')
df_original = pd.read_csv('sales_train.csv')

#lag kopi av orginal dataframe
df_draft = df_original.copy()

# Fjerner nullverdier fra df_concat ved å filtrere dem vekk:
df_draft = df_draft.loc[df_draft['item_cnt_day']>=0]

#sjekk at ingen retur-salg har kommet med
(df_draft['item_cnt_day'] == -1).sum()

# Legger til 'total sales kolonne' 
df_draft['Total_Sales_day'] = df_draft['item_price'] * df_draft['item_cnt_day']

#%%
# filter data for one shop
df_one_month = df_draft.loc[df_draft['date_block_num']== 0, :]
df_grouped_shop = df_one_month[['shop_id', 'date', 'Total_Sales_day']].groupby(by=['shop_id', 'date']).sum().reset_index()

#%%
# One Hot Encoder av shop-id:
shop_encoder = OneHotEncoder(sparse=False)
df_ohe_shops = pd.DataFrame (shop_encoder.fit_transform(df_grouped_shop[['shop_id']]))
df_ohe_shops.columns = shop_encoder.get_feature_names(['shop_id'])

#%%
# Fikser dato
# Gjør om date til datetime-format:
df_grouped_shop['date']= pd.to_datetime(df_grouped_shop['date'],format= '%d.%m.%Y')


# One Hot Encoder av måned:
df_grouped_shop['month'] = df_grouped_shop['date'].dt.month
month_encoder = OneHotEncoder(sparse=False)
df_ohe_months = pd.DataFrame (month_encoder.fit_transform(df_grouped_shop[['month']]))
df_ohe_months.columns = month_encoder.get_feature_names(['month'])

# One Hot Encoder av år:
df_grouped_shop['year'] = df_grouped_shop['date'].dt.year
year_encoder = OneHotEncoder(sparse=False)
df_ohe_years = pd.DataFrame (year_encoder.fit_transform(df_grouped_shop[['year']]))
df_ohe_years.columns = year_encoder.get_feature_names(['year'])

# One Hot Encoder av dag:
df_grouped_shop['day'] = df_grouped_shop['date'].dt.day
day_encoder = OneHotEncoder(sparse=False)
df_ohe_days = pd.DataFrame(day_encoder.fit_transform(df_grouped_shop[['day']]))
df_ohe_days.columns = day_encoder.get_feature_names(['day'])

# One Hot Encoder av kvartal:
df_grouped_shop['quarter'] = df_grouped_shop['date'].dt.quarter
quarter_encoder = OneHotEncoder(sparse=False)
df_ohe_quarter = pd.DataFrame (quarter_encoder.fit_transform(df_grouped_shop[['quarter']]))
df_ohe_quarter.columns = quarter_encoder.get_feature_names(['quarter'])

# One Hot Encoder av ukedag:
df_grouped_shop['dayofweek_text'] = df_grouped_shop['date'].dt.day_name() #Gir i tekst
dayofweek_encoder = OneHotEncoder(sparse=False)
df_ohe_dayofweek = pd.DataFrame (dayofweek_encoder.fit_transform(df_grouped_shop[['dayofweek_text']]))
df_ohe_dayofweek.columns = dayofweek_encoder.get_feature_names(['Day:'])

# Legger til day of week(numerisk) og tekst. 
#NB! Mandag=0, søndag=6
df_grouped_shop['dayofweek'] = df_grouped_shop['date'].dt.dayofweek # Numerisk

# Sjekker om det er helg eller ikke:
# np.where()= hvis sant, gjør x, ellers y
df_grouped_shop['is_weekend']= np.where(df_grouped_shop['dayofweek_text'].isin(['Sunday','Saturday']),1,0)
df_weekend = df_grouped_shop['is_weekend']

#%%
# Skalerer ved log:
np_log_sales = np.log( np.c_[df_grouped_shop['Total_Sales_day']])

df_log_sales=pd.DataFrame(np_log_sales)

# Standardisering:   
sales_standardiser=StandardScaler()   
df_std_log_sales = pd.DataFrame(sales_standardiser.fit_transform(df_log_sales))

#%%
""" Her er preprocesseringen ferdig """
#%%

df_analysis = pd.concat([
    df_ohe_days, 
    df_ohe_dayofweek, 
    df_ohe_months,
    df_ohe_years, 
    df_ohe_quarter, 
    df_weekend, 
    df_ohe_shops,
    df_std_log_sales], 
    axis=1)

#%%


X_one_month = np.c_[df_concat[['shop_id', 'year','month', 'day','quarter','dayofweek', 'is_weekend' ]]]
y_one_month = np.c_[df_concat['Total_Sales_day']]


# Train, test, split:
X_train, X_test, y_train, y_test = train_test_split(
    X_one_month, y_one_month, test_size=0.3, random_state=420)

X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.333, random_state=420)

X_train.reshape(-1,7)
X_test.reshape(-1,7)
X_val.reshape(-1,7)
y_train.reshape(-1,1)
y_test.reshape(-1,1)
y_val.reshape(-1,1)

input_layer = Input(shape=(7,))
first_hidden_layer = Dense(3, activation = "relu")(input_layer)
second_hidden_layer = Dense(3, activation = "relu")(first_hidden_layer)
third_hidden_layer = Dense(2, activation = "relu")(second_hidden_layer)
output_layer = Dense(1, activation = "linear")(third_hidden_layer)

linear_model1 = Model(inputs = input_layer, outputs = output_layer)
linear_model1.compile(optimizer='adam', loss='mse', metrics=['mae'])

linear_model1.fit(X_train, y_train, batch_size=32, epochs=100)



















# prediction = linear_model1.predict(X_one_month)


print(prediction)
#Om vi ønsker å sette inn prediction:
X_new = [[]] 
print(model.predict(X_new))


# Naiv modell:
y_mean = np.mean(y)*np.ones(y.shape)#mean = gjennomsnittet

print(model.score(X = X, y = y))
y_pred = model.predict(X) 
print(np.sqrt(mean_squared_error(y_pred, y)))
print(mean_absolute_error(y_pred, y))


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

