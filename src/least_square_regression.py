from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import csv

raw_train = list(csv.DictReader(open('liberia_train.csv','rU')))
raw_test = list(csv.DictReader(open('liberia_test.csv','rU')))

# train data
X = []
for x in raw_train:
  X.append([float(x['Day'])])
X = np.array(X)
Y = []
for y in raw_train:
    Y.append(y['Value'])
Y = np.array(Y,np.double)

# test data
X_test = []
for x in raw_test:
  X_test.append([float(x['Day'])])
X_test = np.array(X_test)
Y_test = []
for y in raw_test:
    Y_test.append(y['Value'])
Y_test = np.array(Y_test,np.double)

# train
clf = linear_model.LinearRegression()
clf.fit(X,Y,2)
print clf.coef_, clf.intercept_

# test
print np.mean((clf.predict(X_test)-Y_test)**2)

# plot
plt.plot(X,Y,'x',X_test,Y_test,'*',)
x_graph = range(250)
X_graph = []
for x in x_graph:
    X_graph.append([x])
X_graph = np.array(X_graph)
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.plot(X_graph, clf.predict(X_graph),linewidth=2)
plt.show()