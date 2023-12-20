import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

hp_tot_list_list =[[95.0, 90.0, 80.0],[ 95.0, 90.0, 80.0],[ 95.0, 90.0, 80.0],[ 95.0, 90.0, 80.0]]

'''
#per favore notare che predict samples ritorna un hp_list. se noi appendiamo queste hp_list a un'altra lista vuota, otteniamo hp_tot_list_list.
hp1 = [1,2,3,4,5]
hp2 = [6,7,8,9,10]

hptot= []
hptot.append(hp1)
hptot.append(hp2)
print(hptot) #stessa struttura di hp_tot_list_list
'''

data_folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
metadata = pd.read_excel(data_folder + 'metadata2022_04.xlsx')

X = np.array(hp_tot_list_list)
y = np.array(metadata['AHA'].values[:4])

model = LinearRegression()
rkf = RepeatedKFold(n_splits=2, n_repeats=10)
score = cross_val_score(model, X, y, cv=rkf)

print(type(X),'  ', X.shape)
print(type(y),'  ', y.shape)

print('mean score:', score.mean())