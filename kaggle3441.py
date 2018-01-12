#LOAD THE MODULES##  								#THIS ONE FOR LR ONLY

#INPUT THE STATE, MONTH, YEAR

#IF STATE== STATE-1 
#  IF MONTH== 1ST MONTH

###FOR STATE-1 #######
######FOR 1ST MONTH#####
#LOAD THE DATA - 
#YEAR COLUMN-X
#RAINFALL LEVEL- Y

#FIT AND PRINT THE PREDICTION
#####FOR 2ND MONTH#########
#DO THE SAME
###############################
import numpy as np 
import pandas as pd 
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB							
from sklearn.gaussian_process import GaussianProcessClassifier
import csv
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import time

start_time = time.time()
df = pd.read_csv('rainfall in india 1901-2015.csv')             
df = df.fillna(0)
#state = input("ENter the state: ")
month = raw_input("Enter month of rainfall: ") 
year= raw_input("Enter the year for prediction: ")
#print("Here is the predicted level of rainfall for the following state: ")

year1 = year  #will be used later

X = df[month]
X = X[1:110]
X = X.values.reshape([X.shape[0],-1])

Y = df['YEAR']
Y = Y[1:110]

#V = np.sort(5 * np.random.rand(40, 1), axis=0)
#y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))
#print(np.shape(X) )
#print(np.shape(Y) )


#print(np.shape(V) )
rng = np.random.RandomState(1)
q = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()

print(np.shape(q) )
print(np.shape(y) )

#regr_1 = DecisionTreeRegressor(max_depth=120)
#regr_2 = DecisionTreeRegressor(max_depth=5)
#regr_1.fit(X, Y)
#regr_2.fit(X, Y)

# Predict
#X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
#y_1 = regr_1.predict(X)
#y_2 = regr_2.predict(X)

# Plot the results
#plt.figure()
#plt.scatter(X, Y, s=20, edgecolor="black",
#            c="darkorange", label="data")
#plt.plot(X, y_1, color="cornflowerblue",
#         label="max_depth=120", linewidth=2)
#plt.plot(X, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
#plt.xlabel("RAINFALL IN THAT MONTH OVER THE YEARS")
#plt.ylabel("THE YEARS-> 1901-2015")
#plt.title('RAINFALL STATISTICS')
#plt.legend()
#plt.grid(True)
#plt.show(block=False)
#time.sleep(3)
#plt.close()


Y = df[month]
Y = Y[1:110]
#Y = Y.values.reshape([Y.shape[0],-1])
#Y_train = Y[1:70]
#Y_train = Y_train.values.reshape([Y_train.shape[0],-1])

X = df['YEAR']
X = X[1:110]
#X_train = X[1:110]
X = X.values.reshape([X.shape[0],-1])

#V = np.sort(5 * np.random.rand(40, 1), axis=0)
#y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))
#print(np.shape(X) )
#print(np.shape(Y) )


#print(np.shape(V) )
#regr_1 = DecisionTreeRegressor(max_depth=120)
#regr_2 = DecisionTreeRegressor(max_depth=5)
#regr_1.fit(X,Y)
#regr_2.fit(X,Y)

#regr_1.fit(X_train, Y_train)
#regr_2.fit(X_train, Y_train)

# Predict
#X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
#X_test = X[71:110]
#X_test = X_test.values.reshape([X_test.shape[0],-1])

#y_1 = regr_1.predict(year)
#y_2 = regr_2.predict(X_test)

#print("THE PREDICTED RAINFALL IN THE MONTH OF " + month + "IN THE YEAR " + year + "IS: " + y_1)
#print("[SVR]: THE PREDICTED RAINFALL IN THE MONTH OF month IN THE YEAR IS: ")
#print(y_1)
#plt.scatter(X, Y, color='red', label='data')
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=3)
#y_rbf = svr_rbf.fit(X, Y).predict(X)
#y_lin = svr_lin.fit(X, Y).predict(X)
#y_poly = svr_poly.fit(X, Y).predict(X)
#lw = 2
#plt.figure(figsize=(12, 7))
#plt.scatter(X, Y, color='red', label='data')
#plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
#plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
#plt.xlabel("RAINFALL IN THAT MONTH OVER THE YEARS")
#plt.ylabel("THE YEARS-> 1901-2015")
#plt.title('RAINFALL STATISTICS')
#plt.legend()
#plt.grid(True)
#plt.show()


reg=linear_model.LinearRegression() 
reg.fit(X,Y)
a = reg.predict(X)
#print("the expected rainfall in the entered year  is : " + a)

plt.scatter(X, Y, color='red', label='data')
plt.plot(X, Y, color='blue', label='test fitting')
plt.title('[LR]: RAINFALL STATISTICS')
plt.legend()
plt.grid(True)
plt.show(block=False)
time.sleep(3)
plt.close()
print(type(X))
print(type(year))

#TO CNOVERT YEAR-A STRING TO NUMPY ARRAY
year = np.array(list(year))
year= year.reshape(-1,1)
year = year.astype(float)	#convert it to float


reg1=linear_model.LinearRegression() 
reg1.fit(X,Y)                                             
a1 = reg1.predict(year)
print("[LR]: THE PREDICTED RAINFALL IN THE MONTH OF month IN THE YEAR IS: ")
print(np.sum(a1)/40)
gg = np.sum(a1)/40
#for plotting the Y along with prediction
X1 = np.append(X,year1)
opY = np.append(Y,gg)
#opY = opY[1:]
print(opY.shape)
print(X1.shape)
#

plt.scatter(X, Y, color='red', label='data')
plt.plot(X1, opY, color='blue', label='LR fitting')
plt.title('[LR]: RAINFALL STATISTICS')
plt.legend()
plt.grid(True)
plt.show(block=False)
time.sleep(3)
plt.close()
#plt.show(block=False)
#time.sleep(3)
#plt.close()
time.sleep(100)
print("--- %s seconds ---" % (time.time() - start_time))

