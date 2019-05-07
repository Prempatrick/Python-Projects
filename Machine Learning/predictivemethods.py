import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_predict, cross_val_score
import sklearn.metrics

# This Data is usually to test the new predictive techniques, so i will use as basic model amd apply 
# the advance models. So models are as follows

#1. Simple Regression 


# The data is Auto motive data set avaiable in kaggle link: https://www.kaggle.com/uciml/autompg-dataset

data= pd.read_csv(r"F:\Dataset\UCI\Auto\autos_mpg.csv",na_values='?')

data.dtypes

# Checking the columns and shape

print("There are {} variables and {} rows in the Auto dataset".format(data.shape[1],data.shape[0]))


# Checking the missing values 

data.isna().sum()

# There are 6 values missing in the horsepower variable which we will imputing by mean

data['horsepower']=data['horsepower'].fillna(data.horsepower.mean())



# Spiltting into Test and Train Datasets 





#  I will be performing simple regression, but as we see the data there are many 
# categorical variables hence we perform converting into dummy variables


data_dummy= pd.get_dummies(data[['car_name']])


# Dropping the orginal variables 
data.drop(['car_name'], axis=1, inplace=True)

# Joinging the encoded data back to the orginal data
data= data.join(data_dummy)

# Creating milage per litre as target variables 
X= data.copy()
X.drop('mpg',axis=1,inplace=True)
y= data['mpg']

# Creating test and train Datset out of main dataset 

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)


# Regression model 
reg= LinearRegression()

# fitting the model 

model1=reg.fit(X_train,y_train)


# predicting the model 
pred1=model1.predict(X_test)


mse= mean_squared_error(pred1,y_test)
print("The mean squared error is {}".format(mse))

r2= r2_score(pred1,y_test)
print("The R2 score is {}".format(r2))

# Plotting the Actual vs Predicted Plot 

fig,axes=plt.subplots(figsize=(8,8))
axes.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], style='k+')
sns.scatterplot(pred1,y_test)


# Using Kfold validation

model2= cross_val_score(reg,X_train,y_train,cv=5,scoring='r2')




pred2= model2.predict(X_test)









