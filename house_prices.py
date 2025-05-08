import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.linear_model import LinearRegression

data = {
    "Area":[500,1000,1500,2000,2500,3000],
    "Price":[15,30,45,60,75,85]
}
df= pd.DataFrame(data)
x= df[['Area']]
y = df['Price']

model = LinearRegression()
model.fit(x,y)
pred_values = model.predict([[3500]])

print(f"The predicted values are :{pred_values}")

plt.scatter(x,y, color = 'blue')
plt.plot(x,model.predict(x),color = 'red' )
plt.xlabel("Area changes")
plt.ylabel("Prce changes")
plt.show()
