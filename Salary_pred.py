import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

data = {
    "Experience":[1,2,3,4,5,6,7,8,9,10],
    "Salary":[150,200,250,300,350,400,450,500,550,600]
}
df = pd.DataFrame(data)
x = df[['Experience']]
y = df['Salary']

model = LinearRegression()
model.fit(x,y)
pred_values = model.predict([[6.5]])
print(f"The predicted salary for 6.5 years are : {pred_values}")

plt.scatter(x,y,color="blue")
plt.plot(x, model.predict(x), color = 'red')
plt.xlabel("Exerience")
plt.ylabel("Salary")
plt.show()