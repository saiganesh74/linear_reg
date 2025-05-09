import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [2, 5, 10, 17, 26, 37, 50, 65, 82, 101]
} 

df = pd.DataFrame(data)
x = df[['Hours']]
y = df['Marks']

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

pred = model.predict(poly.transform([[6.5]]))
print(f"Predicted values are: {pred}")

# Create smooth range for curve
x_range = np.linspace(min(x)[0], max(x)[0], 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range = model.predict(x_range_poly)

# Plot
plt.scatter(x, y, color='blue')
plt.plot(x_range, y_range, color="red")  # curve
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Polynomial Regression Curve")
plt.show()
