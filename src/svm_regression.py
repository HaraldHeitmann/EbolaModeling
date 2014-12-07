from sklearn import svm
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import csv

raw_train = list(csv.DictReader(open('sierra_leone_train.csv','rU')))
raw_test = list(csv.DictReader(open('sierra_leone_test.csv','rU')))

# train data
X = []
for x in raw_train:
    day = float(x['Day'])
    X.append([day,day**2, day**3, 1/(1+np.exp(day))])
X = np.array(X)
Y = []
for y in raw_train:
    Y.append(y['Value'])
Y = np.array(Y,np.double)

# test data
X_test = []
for x in raw_test:
    day = float(x['Day'])
    X_test.append([day,day**2, day**3, 1/(1+np.exp(day))])
X_test = np.array(X_test)
Y_test = []
for y in raw_test:
    Y_test.append(y['Value'])
Y_test = np.array(Y_test,np.double)

# scale data
scaler = preprocessing.MinMaxScaler()
X_scale = scaler.fit_transform(X)
X_scale_test = scaler.transform(X_test)


# train with CV
C = [0.1,0.3,1,3,10,30,100,300,1000,3000]
epsilon = [0.03,0.1,0.3,1,3,10]
param_grid_rbf = [{'kernel': ['rbf'], 'gamma': [1e-2,3e-2,1e-3,3e-3,1e-4,3e-4],'C': C, 'epsilon': epsilon,'tol':[0.0001]}]
param_grid_linear = [{'kernel': ['linear'], 'C': C,'epsilon': epsilon,'tol':[0.0001]}]
param_grid_poly = [{'kernel': ['poly'], 'C': C, 'degree': [2,3,4,5],'epsilon': epsilon,'tol':[0.0001]}]
param_grid_sigmoid = [{'kernel': ['sigmoid'], 'C': C,'epsilon': epsilon,'tol':[0.0001]}]
clf1 = GridSearchCV(svm.SVR(kernel='linear'), param_grid_linear, cv=10, scoring='mean_squared_error')
clf2 = GridSearchCV(svm.SVR(kernel='poly'), param_grid_poly, cv=10, scoring='mean_squared_error')
clf3 = GridSearchCV(svm.SVR(kernel='rbf'), param_grid_rbf, cv=10, scoring='mean_squared_error')
clf4 = GridSearchCV(svm.SVR(kernel='sigmoid'), param_grid_sigmoid, cv=10, scoring='mean_squared_error')
#clf5 = svm.SVR(kernel='sigmoid')
clf1.fit(X_scale,Y)
clf2.fit(X_scale,Y)
clf3.fit(X_scale,Y)
clf4.fit(X_scale,Y)
#clf5.fit(X_scale,Y)
print clf1.best_estimator_, clf1.best_score_
print clf2.best_estimator_, clf2.best_score_
print clf3.best_estimator_, clf3.best_score_
print clf4.best_estimator_, clf4.best_score_

# test
print 'linear:', np.mean((clf1.predict(X_scale_test)-Y_test)**2)
print 'poly:', np.mean((clf2.predict(X_scale_test)-Y_test)**2)
print 'rbf', np.mean((clf3.predict(X_scale_test)-Y_test)**2)
print 'sigmoid', np.mean((clf4.predict(X_scale_test)-Y_test)**2)
#print 'test', np.mean((clf5.predict(X_scale_test)-Y_test)**2)

# plot
plt.plot(X[:,0],Y,'x',X_test[:,0],Y_test,'*',)
x_graph = range(250)
X_graph = []
for x in x_graph:
    day = x
    X_graph.append([day,day**2,day**3, 1/(1+np.exp(day))])
X_graph = np.array(X_graph)
plt.xlabel('Days')
plt.ylabel('Population')
plt.ylim(ymax=8000)
plt.plot(X_graph[:,0], clf1.predict(scaler.transform(X_graph)),'-y')
plt.plot(X_graph[:,0], clf2.predict(scaler.transform(X_graph)),'--')
plt.plot(X_graph[:,0], clf3.predict(scaler.transform(X_graph)),'-.c')
plt.plot(X_graph[:,0], clf4.predict(scaler.transform(X_graph)),'--g')
#plt.plot(X_graph[:,0], clf5.predict(scaler.transform(X_graph)))
plt.show()