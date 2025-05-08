import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns 
import matplotlib.pyplot as plt 

companies = pd.read_csv('/home/saiganesh/Desktop/Linear_Regression/1000_Companies.csv')
X= companies.iloc[ : , :-1].values
Y = companies.iloc[:, 4].values
companies.head()

labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

print(X)