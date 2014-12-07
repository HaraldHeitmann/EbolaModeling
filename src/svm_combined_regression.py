from sklearn import svm
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import csv

def myfunc():
    raw_train = list(csv.DictReader(open('combined_train.csv','rU')))
    raw_test = list(csv.DictReader(open('combined_test.csv','rU')))
    
    def oneHotEncode(country):
        if country == 'Sierra Leone':
            return [1,0,0]
        if country == 'Liberia':
            return [0,1,0]
        return [0,0,1]
    
    # train data
    X = []
    for x in raw_train:
        day = float(x['Day'])
        country = oneHotEncode(x['Country'])
        X.append([day,day**2, day**3, 1/(1+np.exp(day)),country[0],country[1],country[2]])
    X = np.array(X)
    Y = []
    for y in raw_train:
        Y.append(y['Value'])
    Y = np.array(Y,np.double)
    
    
    # test data
    X_test = []
    for x in raw_test:
        day = float(x['Day'])
        country = oneHotEncode(x['Country'])
        X_test.append([day,day**2, day**3, 1/(1+np.exp(day)),country[0],country[1],country[2]])
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
    C = [100,300,1000,3000,10000]
    C_rbf = [100,3000,10000,300000]
    epsilon = [0.03,0.1,0.3,1,3,10,30,100]
    param_grid_linear = [{'kernel': ['linear'], 'C': C,'epsilon': epsilon,'tol':[0.0001]}]
    param_grid_rbf = [{'kernel': ['rbf'], 'gamma': [1e-2,3e-2,1e-3,3e-3,3e-4],'C': C_rbf, 'epsilon': epsilon,'tol':[0.0001]}]
    param_grid_poly = [{'kernel': ['poly'], 'C': C, 'degree': [2,3,4,5,6,7,8],'epsilon': epsilon,'tol':[0.0001]}]
    clf1 = GridSearchCV(svm.SVR(kernel='linear'), param_grid_linear, cv=10, scoring='mean_squared_error', n_jobs=3)
    clf2 = GridSearchCV(svm.SVR(kernel='poly'), param_grid_poly, cv=10, scoring='mean_squared_error',n_jobs=3)   
    clf3 = GridSearchCV(svm.SVR(kernel='rbf'), param_grid_rbf, cv=10, scoring='mean_squared_error',n_jobs=3)
    clf1.fit(X_scale,Y)
    clf2.fit(X_scale,Y)
    clf3.fit(X_scale,Y)
    
    print clf1.best_estimator_, clf1.best_score_
    print clf2.best_estimator_, clf2.best_score_
    print clf3.best_estimator_, clf3.best_score_
    
    # test
    print 'linear:', np.mean((clf1.predict(X_scale_test)-Y_test)**2)
    print 'poly:', np.mean((clf2.predict(X_scale_test)-Y_test)**2)
    print 'rbf', np.mean((clf3.predict(X_scale_test)-Y_test)**2)
    
    # plot
    plt.figure(figsize=(16, 12), dpi=120)
    plt.plot(X[:,0],Y,'x',X_test[:,0],Y_test,'*',)
    x_graph = range(300)
    X_graph = []
    for x in x_graph:
        day = x
        X_graph.append([day,day**2,day**3, 1/(1+np.exp(day)),1,0,0])
    X_graph = np.array(X_graph)
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.ylim(ymax=8000)
    plt.plot(X_graph[:,0], clf1.predict(scaler.transform(X_graph)),'r-.')
    plt.plot(X_graph[:,0], clf2.predict(scaler.transform(X_graph)),'r-')
    plt.plot(X_graph[:,0], clf3.predict(scaler.transform(X_graph)),'r--')
    
    X_graph = []
    for x in x_graph:
        day = x
        X_graph.append([day,day**2,day**3, 1/(1+np.exp(day)),0,1,0])
    X_graph = np.array(X_graph)
    plt.plot(X_graph[:,0], clf1.predict(scaler.transform(X_graph)),'g-.')
    plt.plot(X_graph[:,0], clf2.predict(scaler.transform(X_graph)),'g-')
    plt.plot(X_graph[:,0], clf3.predict(scaler.transform(X_graph)),'g--')
    
    X_graph = []
    for x in x_graph:
        day = x
        X_graph.append([day,day**2,day**3, 1/(1+np.exp(day)),0,0,1])
    X_graph = np.array(X_graph)
    
    plt.plot(X_graph[:,0], clf1.predict(scaler.transform(X_graph)),'b-.')
    plt.plot(X_graph[:,0], clf2.predict(scaler.transform(X_graph)),'b-')
    plt.plot(X_graph[:,0], clf3.predict(scaler.transform(X_graph)),'b--')
    
    plt.show()

if __name__ == "__main__":
    myfunc()