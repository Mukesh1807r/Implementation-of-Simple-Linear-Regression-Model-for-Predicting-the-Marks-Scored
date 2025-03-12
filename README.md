# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mukesh R 
RegisterNumber: 212224240098 
/*
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())
dataset.info()
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
print(X.shape)
print(Y.shape)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)
print(y_pred)
mse=mean_squared_error(Y_test,y_pred)
print('MSE =',mse)
mae=mean_absolute_error(Y_test,y_pred)
print('MAE =',mae)
rmse = np.sqrt(mse)
import matplotlib.pyplot as plt

plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_test, y_pred, color='silver')
plt.title('Test set(H vs )')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()
*/
```

## Output:


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
