# Implementation of Univariate Linear Regression
## Aim:
To implement univariate Linear Regression to fit a straight line using least squares.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
1.	Get the independent variable X and dependent variable Y.
2.	Calculate the mean of the X -values and the mean of the Y -values.
3.	Find the slope m of the line of best fit using the formula.
 ![eqn1](./eq1.jpg)
4.	Compute the y -intercept of the line by using the formula:
![eqn2](./eq2.jpg)  
5.	Use the slope m and the y -intercept to form the equation of the line.
6.	Obtain the straight line equation Y=mX+b and plot the scatterplot.
## Program
```
Developed By:Gokulan.S.S
Register No:26018286
program-1
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the dataset
data = datasets.load_diabetes()
X = data.data
y = data.target

# Split training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model
reg.fit(X_train, y_train)

# Regression coefficients
print("Coefficients:", reg.coef_)

# Variance score
print("Variance score: {}".format(reg.score(X_test, y_test)))

# Plot residual errors
plt.style.use('fivethirtyeight')

# Training data residuals
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green", s=10, label="Train data"
)

# Testing data residuals
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue", s=10, label="Test data"
)

# Zero residual error line
plt.hlines(y=0, xmin=0, xmax=500, linewidth=2)

# Legend and title
plt.legend(loc="upper right")
plt.title("Residual errors")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")

plt.show()

program-2


import numpy as np
import matplotlib.pyplot as plt

# Preprocessing input data
X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

plt.scatter(X, Y)
plt.show()

# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)

num = 0
den = 0

for i in range(len(X)):
    num += (X[i] - X_mean) * (Y[i] - Y_mean)
    den += (X[i] - X_mean) ** 2

m = num / den
c = Y_mean - m * X_mean

print(m, c)

# Making predictions
Y_pred = m * X + c
print(Y_pred)

plt.scatter(X, Y)  # Actual
plt.scatter(X, Y_pred, color="red")
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color="red")  # Predicted
plt.show()






```
## Output
</br><img width="570" height="847" alt="image" src="https://github.com/user-attachments/assets/7e1f491b-0121-40b8-a00e-43b20a8c6f24" />

</br><img width="607" height="445" alt="image" src="https://github.com/user-attachments/assets/ede3e30d-c845-4bbe-9a9a-afd9ebf672ce" />


</br><img width="697" height="756" alt="image" src="https://github.com/user-attachments/assets/c39f7554-447e-483b-8972-5fddee610de3" />
</br><img width="555" height="822" alt="image" src="https://github.com/user-attachments/assets/55752946-3fba-4005-8854-636b1f2316ee" />


## Result
Thus the univariate Linear Regression was implemented to fit a straight line using least squares.
