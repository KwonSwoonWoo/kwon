import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('BIGDATA.csv')
df = df.dropna()

featurename = list(df.columns)
print(featurename)

featurename.remove('YS')
featurename.remove('HRc')
featurename.remove('J')
featurename.remove('El')

total_data = df.loc[:,:'UTS']
y = df.loc[:,'UTS']
del total_data['UTS']
x = total_data


#define dictionary to store ranking
# rank={}

# def ranking(ranks, names, order=1):
#     minmax = MinMaxScaler()
#     ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
#     ranks = map(lambda x: round(x,2),ranks)
#     return dict(zip(names,ranks))



rf = RandomForestRegressor(n_jobs=-1,n_estimators=50,verbose=3)
model = rf.fit(x,y)
# ranks['UTS'] = ranking(rf.feature_importances_,names=names);
pred = rf.predict(x)
print(pred)



feature_importances = model.feature_importances_
ft_importances = pd.Series(feature_importances, index = x.columns)
ft_importances = ft_importances.sort_values(ascending=False)
plt.figure(figsize=(12,10))
sns.barplot(x = ft_importances, y=x.columns)
plt.show()