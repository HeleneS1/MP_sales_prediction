""" This is the file where we connect all the dots """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data= pd.read_csv('sales_train.csv')

#Lager kopi av datasettet
df_draft= data.copy()


# Legger til ny kolonne:
df_draft['Total_Sales_day']=df_draft['item_price']* df_draft['item_cnt_day']


# Sjekker fordelingen:

df_draft.hist()