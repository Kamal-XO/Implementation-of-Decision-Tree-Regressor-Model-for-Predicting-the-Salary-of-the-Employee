# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KAMALESH SV
RegisterNumber:  212222240041
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:

### Data.Head():
![OUTPUT](https://user-images.githubusercontent.com/120204455/279486017-17219e1b-9545-45e2-bb45-dd459016cbf9.png)

### Data.info():
![OUTPUT](https://user-images.githubusercontent.com/120204455/279486035-6c499887-944a-476d-b365-f406cc541e6f.png)

### isnull() and sum():
![OUTPUT](https://user-images.githubusercontent.com/120204455/279486050-e97ab81f-f8b9-4813-83de-327da3214afe.png)

### Data.Head() for salary:
![OUTPUT](https://user-images.githubusercontent.com/120204455/279486068-ffc344dd-39b6-4370-9282-468f4642736c.png)

### MSE Value:
![OUTPUT](https://user-images.githubusercontent.com/120204455/279486086-d063c559-f82f-4a52-b1fd-74c153c7d36e.png)

### r2 Value:
![OUTPUT](https://user-images.githubusercontent.com/120204455/279486100-2956ebf4-c1b2-4a45-9365-21f67717ebc4.png)

### Data Prediction:
![OUTPUT](https://user-images.githubusercontent.com/120204455/279486119-516cbe0b-9937-4dd6-a5a8-1ac01a6673eb.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
