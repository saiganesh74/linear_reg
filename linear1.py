import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

data = {
    "Hours" : [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Marks": [10, 20, 30, 40, 50, 55, 65, 75, 85]    
}

df = pd.DataFrame(data)
x = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(x,y)
pred_marks = model.predict([[7.5]])
print(f"Predicted marks for 7.5 hours is {pred_marks}")
plt.scatter(x,y)
plt.plot(x , model.predict(x),color= "red")
plt.xlabel("Hours studied")
plt.ylabel("Marks attained ")
plt.show()