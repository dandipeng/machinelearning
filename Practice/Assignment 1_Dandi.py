
# HW 1 ECS 171 Dandi Peng

import os
import pandas as pd
import numpy as np
from numpy.linalg import inv

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns



# Read the data
cars = pd.read_csv('auto-mpg.data.txt', sep = "\s+", header= None, 
                   names = ["mpg", "cylinders", "displacement", "horsepower", "weight", 
                            "acceleration", "model_year", "origin", "car_name"])
# convert the "?" to Nan
cars["horsepower"] = pd.to_numeric(cars["horsepower"], errors='coerce')
# drop the Nan
cars = cars.dropna().reset_index(drop = True)


cars.shape

# split by the sorted values
mpg_levels = np.array_split(np.sort(cars.iloc[:,0].values),3)
print("The span of low mpg is", mpg_levels[0][0], ":", (mpg_levels[0][-1]+mpg_levels[1][0])/2)
print("The span of medium mpg is", (mpg_levels[0][-1]+mpg_levels[1][0])/2, ":", mpg_levels[1][-1])
print("The span of high mpg is", mpg_levels[2][0], ":", mpg_levels[2][-1])

# assign a categorical column the df
cars['mpg_level'] = np.where(cars.mpg<=18.6, "low",
                           np.where(cars.mpg < 27.0,"medium", "high"))


# check the type of every columns
cars.dtypes

cars.head()
# plot
sns.catplot(x = "mpg_level", y="mpg", kind="swarm", data=cars)





## Q2
sns.set(style="ticks")

sns.pairplot(cars[cars.columns.difference(["mpg","car_name"])],
             hue="mpg_level", palette="husl", diag_kind = "hist",
             diag_kws = {"color": 'lightblue', 'edgecolor':'lightblue',"linewidth":1})


## Q3
class LinearRegression:
    """
    : A regression object that uses OLS to fit and predict
    : Init: assign the order
    : fit: precompute the beta
    : predict: Return y hat for X_test
    : change: change the order
    """
    def __init__(self, order):
        """
        Store the order
        """
        self.order = order
    
    def fit(self,x,y):
        self.x1 = np.array([x]).T
        self.y = np.array([y]).T
        #
        # get X matrix
        self.X = self.x1 ** 0
        if self.order != 0:
            for i in range(1, self.order+1):
                self.X = np.append(self.X, self.x1 ** i,axis = 1)
        #
        # get beta vector
        self.beta = np.matmul(np.matmul(inv(np.matmul(self.X.T,self.X)), self.X.T),self.y)
        #
        # get y_hat
        y_hat = np.matmul(self.X, self.beta)[:,0]
        return y_hat
    
    def predict(self,x_test):
        self.x1_test = np.array([x_test]).T
        #
        # get X_test matrix
        self.X_test = self.x1_test ** 0
        if self.order != 0:
            for i in range(1, self.order+1):
                self.X_test = np.append(self.X_test, self.x1_test ** i,axis = 1)
        # get y_predict
        y_predict = np.matmul(self.X_test, self.beta)[:,0]
        return y_predict
        
    def change_order(self,order):
        """
        : After changing the order, the new regressor object should fit in the new order
        """
        self.order = order
        print("The order changes to be "+ self.order+". Please remember to refit the model.")





## Q4
"""
Training and Test set split
Split the data into training set(200/392) and testing set(192/392)
"""
from sklearn.model_selection import train_test_split

train, test = train_test_split(cars, test_size = 192/392, random_state = 1)

## get the train and test data set
X_tr = train[train.columns.difference(['mpg','car_name','mpg_level'])]
y_tr = train.loc[:, 'mpg']
y_tr_log = train.loc[:, 'mpg_level']

X_te = test[test.columns.difference(['mpg','car_name','mpg_level'])]
y_te = test.loc[:, 'mpg']
y_te_log = test.loc[:, 'mpg_level']



def fit_predict(name):
    MSE_tr = []
    MSE_te = [name]
    df_pred = pd.DataFrame({name:X_te[name], 'y': y_te}).reset_index(drop = True)
    df_pred["order"]="test data"
    for order in range(4):
        OLS = LinearRegression(order)
        y_hat = OLS.fit(X_tr[name],y_tr)
        y_pred = OLS.predict(X_te[name])
        mse_tr = np.average((y_tr.values-y_hat)**2)
        mse_te = np.average((y_te.values-y_pred)**2)
        MSE_tr.append(mse_tr)
        MSE_te.append(mse_te)
        df_temp = pd.DataFrame({name:X_te[name], 'y': y_pred}).reset_index(drop = True)
        df_temp["order"] = "order: "+str(order)
        df_pred = pd.concat([df_pred, df_temp], ignore_index=True)
    
    return MSE_tr, MSE_te, df_pred



f, ax = plt.subplots(4,2,figsize=(12,24))
mse_tr = []
mse_te = []
position =[]
# get a list of ax indexes
for i in range(0,4):
    for j in range(0,2):
        position.append(i)
        position.append(j)
        

for i in range(0,len(X_tr.columns)):
    mse_tr_temp, mse_te_temp, df_pred = fit_predict(X_tr.columns[i])
    mse_tr.append(mse_tr_temp)
    mse_te.append(mse_te_temp)
    sns.scatterplot(x=X_tr.columns[i], y="y",
                    data=df_pred[df_pred.order == "test data"], 
                    ax=ax[position[2*i],position[2*i+1]])
    sns.lineplot(x=X_tr.columns[i], y="y", hue="order",
                    data=df_pred[df_pred.order != "test data"], 
                 ax=ax[position[2*i],position[2*i+1]])



# MSE for training data
df_mse_tr = pd.DataFrame(mse_tr)
df_mse_tr = df_mse_tr.transpose()
df_mse_tr.columns = df_mse_tr.iloc[0]
df_mse_tr = df_mse_tr[1:].reset_index(drop = True)
df_mse_tr.head()



# MSE for test data
df_mse_te = pd.DataFrame(mse_te)
df_mse_te = df_mse_te.transpose()
df_mse_te.columns = df_mse_te.iloc[0]
df_mse_te = df_mse_te[1:].reset_index(drop = True)
df_mse_te.head()




## Q5
class LinearRegression_X:
    """
    : A regression object that uses OLS to fit and predict
    : Init: assign the order
    : fit: precompute the beta
    : predict: Return y hat for X_test
    : change: change the order
    """
    def __init__(self, order):
        """
        Store the order
        """
        self.order = order
    
    def fit(self,X,y):
        self.y = y
        #
        # get X matrix
        self.X = np.ones(shape = (X.shape[0],1))
        if self.order !=0:
            for i in range(1, self.order + 1):
                self.X = np.append(self.X, X ** i,axis = 1)
        #
        # get beta vector
        self.beta = np.matmul(np.matmul(inv(np.matmul(self.X.T,self.X)), self.X.T),self.y)
        #
        # get y_hat
        y_hat = np.matmul(self.X, self.beta)
        return y_hat
    
    def predict(self, X_test):
        # get X matrix
        self.X_test = np.ones(shape = (X_test.shape[0],1))
        if self.order !=0:
            for i in range(1, self.order + 1):
                self.X_test = np.append(self.X_test, X_test ** i,axis = 1)
        # get y_predict
        y_predict = np.matmul(self.X_test, self.beta)
        return y_predict
        
    def change_order(self,order):
        """
        : After changing the order, the new regressor object should fit in the new order
        """
        self.order = order
        print("The order changes to be "+ self.order + ". Please remember to refit the model.")




MSE_tr_X = []
MSE_te_X = []
for i in range(3):
    OLS_X = LinearRegression_X(i)
    y_hat = OLS_X.fit(X_tr.values,y_tr.values)
    y_pred = OLS_X.predict(X_te.values)
    mse_tr_X = np.average((y_tr.values-y_hat)**2)
    mse_te_X = np.average((y_te.values-y_pred)**2)
    MSE_tr_X.append(mse_tr_X)
    MSE_te_X.append(mse_te_X)



# show the df of MSE_train and MSE_test
pd.DataFrame({'MSE_train':MSE_tr_X, 'MSE_test':MSE_te_X})





## Q6
from sklearn.linear_model import LogisticRegression
Logreg = LogisticRegression()
Logreg.fit(X_tr.values,y_tr_log.values)




y_pred_log = Logreg.predict(X_te.values)



print('Accuracy of logistic regression classifier on training set: {:.10f}'.
      format(Logreg.score(X_tr.values, y_tr_log.values)))
print('Accuracy of logistic regression classifier on test set: {:.10f}'.
      format(Logreg.score(X_te.values, y_te_log.values)))


## Q7

X_new = np.array([[9,6,350,180,80,1,3700]])
OLS_X = LinearRegression_X(2)
OLS_X.fit(X_tr.values,y_tr.values)
OLS_X.predict(X_new)



Logreg.predict(X_new)



## Q8
X_ox = np.array([0.576])


OLS_ox = LinearRegression(2)
OLS_ox.fit(X_tr['horsepower'],y_tr)
OLS_ox.predict(X_ox)

