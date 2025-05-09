import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

doc = pd.read_csv('/home/saiganesh/ml_pros/multivar.csv')
x = doc[["Hours","Sleep","Practice"]]
y = doc["Marks"]

model = LinearRegression()
model.fit(x,y)
pred_vals = model.predict([[6,7,6.5]])
print(f"The predicted marks are {pred_vals}")

y_pred = model.predict(x)

plt.scatter(y, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.grid(True)
plt.show()
