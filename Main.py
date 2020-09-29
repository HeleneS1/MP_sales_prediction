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
#%%
os.chdir('C:/Users/Helene Stabell/Desktop/Academy/Uke 9MP/')
df_original = pd.read_csv('sales_train.csv')

#Lager kopi av datasettet
df_draft = df_original.copy()
df = df_draft[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']]

# Legger til ny kolonne:
df['Total_Sales_day'] = df['item_price'] * df['item_cnt_day']

# Sjekker fordelingen:
# df_draft.hist()


#%%
# filter data for one shop
df_one_month = df.loc[df['date_block_num']== 0, :]
df_grouped_shop = df_one_month[['shop_id', 'date', 'Total_Sales_day']].groupby(by=['shop_id', 'date']).sum().reset_index()

#%%
# linear_model = LinearRegression()
# linear_model.fit(X = X_one_month, y = y_one_month)

# for row in df_grouped_shop.iterrows():
#     date = row[1]['date']
#     string = str(date)
#     date_clean = ''
#     for letter in string:
#         if letter.isnumeric():
#             letter.join(date_clean)
#     print(date_clean)



#%%

X_one_month = np.c_[df_grouped_shop[['shop_id', 'date']]]
y_one_month = np.c_[df_grouped_shop['Total_Sales_day']]


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

