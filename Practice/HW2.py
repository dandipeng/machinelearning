
# coding: utf-8

# In[2]:


import os
from scipy import stats
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy.linalg import inv

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns


# In[3]:


from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# In[4]:


protein = pd.read_csv('yeast.data.txt', header =None, sep = "\s+",
                     names = ["sequence", "mcg", "gvh", "alm", "mit", "erl",
                      "pox", "vac", "nuc", "class"])


# In[5]:


protein.head()


# In[6]:


protein.shape


# In[7]:


protein.groupby(['class']).size()


# ### 1. Outlier detection algorithms

# In[8]:


outliers_fraction = 0.1
classifiers = [
    ("One-Class SVM", svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf")),
    ("Isolation Forest", IsolationForest(max_samples=len(protein.index),
                                        contamination=outliers_fraction)),
    ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=5,
                                               algorithm = 'auto',
                                               contamination = outliers_fraction))]

X = protein[protein.columns.difference(['sequence','class'])]
y = protein.loc[:, 'class']

# try different outlier detection algorithms (oda)
for oda_name, oda in classifiers:
    # fit the data and tag outliers
    if oda_name == "Local Outlier Factor":
        y_pred = oda.fit_predict(X)
        scores_pred = oda.negative_outlier_factor_
    else:
        oda.fit(X)
        scores_pred = oda.decision_function(X)
        y_pred = oda.predict(X)
    threshold = stats.scoreatpercentile(scores_pred,
                                        100 * outliers_fraction)
    n_errors = y_pred[y_pred == -1].size
    print("By "+oda_name+", there are "+str(n_errors)+" outliers.")


# __(a)__ From the above output, first, as the real data set shows, one class `ERL` only has 5 samples, therefore, I set the n_neighbors = 5 when applying _LoF*_method, and all methods show that there are outliers.
#
# __(b)__ Three of four methods show that there are **149** outliers. They are  _LOF_ and _Isolation Forest_.
#
# __(c)__ By checking the dataset aside of outliers selected through different methods, the _LOF_ keeps the `ERL` class. Thus, I used _LOF_ to remove outliers as the following.

# In[9]:


# Remove 149 outliers
protein_new = protein[y_pred != -1].reset_index(drop=True)
protein_new.shape


# In[10]:


protein_new.groupby(['class']).size()


# ### 2. ANN

# In[11]:


"""
Training and Test set split
Split the data into training set(200/392) and testing set(192/392)
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE = LabelEncoder()
protein_new['y'] = LE.fit_transform(protein_new['class'])

train, test = train_test_split(protein_new, test_size = 0.3, random_state = 1)

## get the train and test data set
X_tr = train[train.columns.difference(['sequence','class','y'])]
y_tr = train.loc[:, 'y']

X_te = test[test.columns.difference(['sequence','class','y'])]
y_te = test.loc[:, 'y']

## get the train and test data set for "CYT"
cyt_x_tr = X_tr.loc[train['class'] == "CYT"]
cyt_x_te = X_te.loc[test['class'] == "CYT"]


# In[12]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)


# In[14]:


onehotencoder = OneHotEncoder()
y_tr_class = onehotencoder.fit_transform(np.array([y_tr]).T).toarray()
y_te_class = onehotencoder.fit_transform(np.array([y_te]).T).toarray()

cyt_y_tr_class = y_tr_class[train['class'] == "CYT"]
cyt_y_te_class = y_te_class[test['class'] == "CYT"]


# In[15]:


# Importing the Keras libraries and packages
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import LambdaCallback


# In[16]:


model = Sequential()
model.add(Dense(3, input_dim= X_tr.shape[1], kernel_initializer='random_uniform',
                bias_initializer='random_uniform',activation='sigmoid'))
model.add(Dense(3, activation = "sigmoid"))
model.add(Dense(10, activation = "softmax"))
model.summary()


# In[17]:


model.compile(optimizer='sgd',
              loss='categorical_crossentropy', metrics = ['accuracy'])


# In[18]:


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []
        self.cyt_tr_hist = []
        self.cyt_te_hist = []

    def on_epoch_end(self, epoch, logs={}):
        self.weights.append(model.layers[2].get_weights())
        self.cyt_tr_hist.append(model.evaluate(cyt_x_tr,cyt_y_tr_class)[0])
        self.cyt_te_hist.append(model.evaluate(cyt_x_te,cyt_y_te_class)[0])


history = LossHistory()

model_hist = model.fit(X_tr, y_tr_class, validation_data=(X_te,y_te_class),
                  batch_size = 1, nb_epoch = 200,callbacks = [history])


# In[19]:


CYT_his = []
for record in history.weights:
    CYT_his.append([record[1][0],record[0][0][0],record[0][1][0],record[0][2][0]])

CYT_w = DataFrame.from_records(CYT_his,columns=["bias","w1","w2","w3"])


# In[20]:


CYT_w.head()


# In[25]:


CYT_w.plot(figsize=(10,10), grid=True)
plt.rcParams.update({'font.size': 22})
plt.title('Weights change')
plt.ylabel('weights')
plt.xlabel('epoch')


# In[24]:


# summarize history for loss
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 22})
plt.plot(history.cyt_tr_hist)
plt.plot(history.cyt_te_hist)
#plt.ylim((1,1.20))
plt.title('CYT loss')
plt.ylabel('CYT loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# ### 3. Retrain

# In[26]:


# create the dummy variable for "class"
protein['y'] = LE.fit_transform(protein['class'])
# X = protein.as_matrix(columns=protein.columns[1:-1])
X = protein.iloc[:,1:-2].values
y = protein.iloc[:,-1].values
y_class = onehotencoder.fit_transform(np.array([y]).T).toarray()


# In[32]:


model_all = Sequential()
model_all.add(Dense(3, input_dim= X_tr.shape[1], kernel_initializer='random_uniform',
                bias_initializer='random_uniform',activation='sigmoid'))
model_all.add(Dense(3, activation = "sigmoid"))
model_all.add(Dense(10, activation = "softmax"))


# In[33]:


class LossHistoryAll(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []

    def on_epoch_end(self, epoch, logs={}):
        self.weights.append(model.layers[2].get_weights())

model_all.compile(optimizer='sgd',
              loss='categorical_crossentropy', metrics = ['accuracy'])


history_all = LossHistoryAll()

model_all_hist = model.fit(X, y_class,
                  batch_size = 1, nb_epoch = 200,callbacks = [history_all])


# In[34]:


history_all.weights[-1]


# In[94]:


1-model_all_hist.history['acc'][-1]


# ### 4. Hand calculation

# In[43]:


X_1 = np.array([X[0]])


# In[44]:


y_1 = np.array([y_class[0]])


# In[69]:


hand_1 = Sequential()
hand_1.add(Dense(3, input_dim= 8, weights=[np.ones((8,3)),np.zeros(3)],activation='sigmoid'))
hand_1.add(Dense(3, weights=[np.ones((3,3)),np.zeros(3)], activation = "sigmoid"))
hand_1.add(Dense(10, weights=[np.ones((3,10)),np.zeros(10)], activation = "softmax"))
hand_1.summary()


# In[70]:


sgd = SGD(lr=0.5)
hand_1.compile(optimizer=sgd,
              loss='categorical_crossentropy', metrics = ['accuracy'])
hand_1_hist = hand_1.fit(X_1, y_1, batch_size = 1, nb_epoch = 1)


# In[71]:


hand_1.get_weights()


# In[64]:


hand_1.layers[1].get_weights()


# In[51]:


hand_1_hist.layers


# ### 5. Grid search

# In[72]:


grid = []
for i in range(1,4):
    for j in range(3,13,3):
        grid.append([i,j])
grid = np.array(grid)
grid = np.reshape(np.reshape(grid, grid.shape + (1,)),(3,4,2))
grid.shape


# In[73]:


grid


# In[86]:


def ANN(hidden_layer,nodes,X_tr,y_tr,X_te,y_te,epoch):
    model = Sequential()
    model.add(Dense(nodes,activation = "sigmoid", input_dim=8))
    for l in range(hidden_layer-1):
        model.add(Dense(nodes, activation = "sigmoid"))
    model.add(Dense(10, activation = "softmax"))

    model.compile(optimizer='sgd',
              loss='categorical_crossentropy', metrics = ['accuracy'])
    model_hist = model.fit(X_tr, y_tr_class,validation_data=(X_te,y_te_class),
                           batch_size = 1, nb_epoch = epoch)
    fnl_loss = model_hist.history['val_loss'][-1]

    return fnl_loss


# In[87]:


fnl_loss = {}
for i in range(3):
    for j in range(4):
        loss = ANN(grid[i,j][0],grid[i,j][1],X_tr,y_tr_class,X_te,y_te_class,100)
        fnl_loss.update({str(grid[i,j]): loss})


# In[88]:


fnl_loss


# From the above output, by comparing the losses, when there is 1 hidden layer with 4 nodes, the loss is the smallest.

# ## 6. Prediction

# In[78]:


X_fnl = protein_new.iloc[:,1:-2].values
y_fnl = protein_new.iloc[:,-1].values
y_fnl_class = onehotencoder.fit_transform(np.array([y_fnl]).T).toarray()


# In[89]:


model_fnl = Sequential()
model_fnl.add(Dense(12, input_dim= X_fnl.shape[1], kernel_initializer='random_uniform',
                bias_initializer='random_uniform',activation='sigmoid'))
model_fnl.add(Dense(10, activation = "softmax"))
model_fnl.compile(optimizer='sgd',
              loss='categorical_crossentropy', metrics = ['accuracy'])


# In[90]:


model_fnl_hist = model_fnl.fit(X_fnl, y_fnl_class,
                  batch_size = 1, nb_epoch = 200)


# In[91]:


# new instance where we do not know the answer
Xnew = np.array([[0.52,0.47,0.52,0.23,0.55,0.03,0.52,0.39]])
#Xnew = sc.transform(Xnew)
# make a prediction
ynew = model_fnl.predict_classes(Xnew)


# In[92]:


protein.loc[protein['y'] == ynew[0], 'class'].iloc[0]
