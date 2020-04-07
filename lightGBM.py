import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier, plot_importance

df = pd.read_csv('BIGDATA.csv')
df = df.dropna()

featurename = list(df.columns)
#print(featurename)

featurename.remove('YS')
featurename.remove('HRc')
featurename.remove('J')
featurename.remove('El')

total_data = df.loc[:,:'UTS']
y = df.loc[:,'UTS']
del total_data['UTS']
x = total_data


lgb = LGBMClassifier(n_estimators=100)
evals = [(x,y)]
lgb.fit(x,y,eval_set=evals,verbose=True)
















