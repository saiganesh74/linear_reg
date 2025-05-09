import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass':  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
x= df[['Hours']]
y = df['Pass']

model = LogisticRegression()
model.fit(x,y)
prob = model.predict_proba([[12]])
pred = model.predict([[12]])

print(f"Probability of passng the exam is :{prob}")
print(f"Prediction is {'Pass' if pred[0]==1 else 'fail'}")

x_range = np.linspace(0,12,100).reshape(-1,1)
y_prob = model.predict_proba(x_range)[:, 1]

plt.plot(x_range, y_prob, color='red', label='Sigmoid Curve')

plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Hours vs Pass')
plt.grid(True)


plt.show()
