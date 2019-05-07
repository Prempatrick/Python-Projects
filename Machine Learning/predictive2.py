import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

fruits=pd.read_csv("https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt", delimiter="\t")


#There are four types of fruits 

fruits.fruit_name.value_counts()

'''
orange      19
apple       19
lemon       16
mandarin     5
'''

# plotting the same 

sns.countplot(fruits['fruit_name'])

# Plotting the box plots for the 

fruits.drop('fruit_name', axis=1). plot(kind='box', layout=(3,3),sharex= False, sharey=False, subplots=True, figsize=(9,9))


sns.pairplot(fruits.drop('fruit_name', axis=1))




# To build model we create target variables and predictors 

feature_names= ['mass', 'width', 'height', 'color_score']

X= fruits[feature_names]
y= fruits['fruit_label']

X.describe()

X_train,X_test,y_train, y_test= train_test_split(X,y,test_size=0.2)


scaler= MinMaxScaler()

X_train= scaler.fit_transform(X_train)
X_test= scaler.fit_transform(X_test)




# Building Logistic Regression 

from sklearn.linear_model import LogisticRegression
 

logreg= LogisticRegression()

modellog1= logreg.fit(X_train,y_train)
modellog1.score(X_train,y_train)


# Implementing the Decision Tree 

from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier()

modelclass1= clf.fit(X_train,X_test)

modelclass1.fit()













