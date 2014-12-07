from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import csv

raw_train = list(csv.DictReader(open('guinea.csv','rU')))
raw_test = list(csv.DictReader(open('guinea.csv','rU')))

# train data
X = []
for x in raw_train:
    day = float(x['Day'])
    X.append([day, day**2, day**3, 1/(1+np.exp(day))])
X = np.array(X)
Y = []
for y in raw_train:
    Y.append(y['Value'])
Y = np.array(Y,np.double)

# test data
X_test = []
for x in raw_test:
    day = float(x['Day'])
    X_test.append([day, day**2, day**3, 1/(1+np.exp(day))])
X_test = np.array(X_test)
Y_test = []
for y in raw_test:
    Y_test.append(y['Value'])
Y_test = np.array(Y_test,np.double)

# train
cvParams = [0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,1,3]
clf3 = linear_model.ElasticNetCV(max_iter=2000,eps=0.0001)
clf3.fit(X,Y)
print 'ElasticNet Guinea:', clf3.coef_, clf3.intercept_, clf3.alpha_, np.min(clf3.mse_path_)

# test
print 'ElasticNet Guinea:', np.mean((clf3.predict(X_test)-Y_test)**2)
print clf3.predict([310,310**2,310**3,1/(1+np.exp(310))])

# plot
plt.figure(figsize=(16, 12), dpi=120)
plt.plot(X[:,0],Y,'rx')
x_graph = range(330)
X_graph = []
for x in x_graph:
    X_graph.append([x,x**2,x**3,1/(1+np.exp(day))])
X_graph = np.array(X_graph)
plt.xlabel('Days')
plt.ylabel('Population')
plt.plot(x_graph, clf3.predict(X_graph), 'r--')

###################################################################################

raw_train = list(csv.DictReader(open('liberia.csv','rU')))
raw_test = list(csv.DictReader(open('liberia.csv','rU')))

# train data
X = []
for x in raw_train:
    day = float(x['Day'])
    X.append([day, day**2, day**3, 1/(1+np.exp(day))])
X = np.array(X)
Y = []
for y in raw_train:
    Y.append(y['Value'])
Y = np.array(Y,np.double)

# test data
X_test = []
for x in raw_test:
    day = float(x['Day'])
    X_test.append([day, day**2, day**3, 1/(1+np.exp(day))])
X_test = np.array(X_test)
Y_test = []
for y in raw_test:
    Y_test.append(y['Value'])
Y_test = np.array(Y_test,np.double)

# train
cvParams = [0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,1,3]
clf3 = linear_model.ElasticNetCV(max_iter=2000,eps=0.0001)
clf3.fit(X,Y)
print 'ElasticNet Liberia:', clf3.coef_, clf3.intercept_, clf3.alpha_, np.min(clf3.mse_path_)

# test
print 'ElasticNet Liberia:', np.mean((clf3.predict(X_test)-Y_test)**2)
print clf3.predict([310,310**2,310**3,1/(1+np.exp(310))])
plt.plot(x_graph, clf3.predict(X_graph), 'g-')
plt.plot(X[:,0],Y,'gx')
x_graph = range(330)
X_graph = []
for x in x_graph:
    X_graph.append([x,x**2,x**3,1/(1+np.exp(day))])
X_graph = np.array(X_graph)

##################################################################################

raw_train = list(csv.DictReader(open('sierra_leone.csv','rU')))
raw_test = list(csv.DictReader(open('sierra_leone.csv','rU')))

# train data
X = []
for x in raw_train:
    day = float(x['Day'])
    X.append([day, day**2, day**3, 1/(1+np.exp(day))])
X = np.array(X)
Y = []
for y in raw_train:
    Y.append(y['Value'])
Y = np.array(Y,np.double)

# test data
X_test = []
for x in raw_test:
    day = float(x['Day'])
    X_test.append([day, day**2, day**3, 1/(1+np.exp(day))])
X_test = np.array(X_test)
Y_test = []
for y in raw_test:
    Y_test.append(y['Value'])
Y_test = np.array(Y_test,np.double)

# train
cvParams = [0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,1,3]
clf3 = linear_model.ElasticNetCV(max_iter=2000,eps=0.0001)
clf3.fit(X,Y)
print 'ElasticNet Sierra Leone:', clf3.coef_, clf3.intercept_, clf3.alpha_, np.min(clf3.mse_path_)

# test
print 'ElasticNet Sierra Leone:', np.mean((clf3.predict(X_test)-Y_test)**2)
print clf3.predict([310,310**2,310**3,1/(1+np.exp(310))])
plt.plot(X[:,0],Y,'bx')
x_graph = range(330)
X_graph = []
for x in x_graph:
    X_graph.append([x,x**2,x**3,1/(1+np.exp(day))])
X_graph = np.array(X_graph)
plt.plot(x_graph, clf3.predict(X_graph), 'b-.')
plt.show()