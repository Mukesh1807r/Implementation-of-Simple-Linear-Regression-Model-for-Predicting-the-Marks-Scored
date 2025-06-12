# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mukesh R 
RegisterNumber: 212224240098

```

```python
---------------------------------------------------------------------------------------------------------------------------------------------------------------

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
```

## Output:
![image](https://github.com/user-attachments/assets/19ac9d7a-0d86-4081-bdcb-9bcbb50a5337)

![image](https://github.com/user-attachments/assets/5ff6a529-0096-4296-a3d3-586eed6ae0a9)

![image](https://github.com/user-attachments/assets/5682ff45-e898-4aa5-9a74-f3ed0b582493)

![image](https://github.com/user-attachments/assets/994e9cf7-759d-4941-b82a-ec5a8693e51c)

![image](https://github.com/user-attachments/assets/e41633ab-0da4-4083-a4f3-a03ab00decea)

![image](https://github.com/user-attachments/assets/f244eb41-6b32-44b1-ae8b-c771d1b9f9e1)

![image](https://github.com/user-attachments/assets/563cae76-333b-43bf-8d07-922d7abf9df2)

![image](https://github.com/user-attachments/assets/c52d646d-3caf-4d76-8580-d605c5d1d907)










## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
