import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df= pd.read_csv('')

X = np.c_[df[""]]
y = np.c_[df[""]]

linear_model = LinearRegression()
linear_model.fit(X = X, y = y)


#%%

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

input_layer = Input(shape=(4,))
first_hidden_layer = Dense(3, activation = "relu")(input_layer)
second_hidden_layer = Dense(3, activation = "relu")(first_hidden_layer)
third_hidden_layer = Dense(2, activation = "relu")(second_hidden_layer)
output_layer = Dense(1, activation = "linear")(third_hidden_layer)


#%%


