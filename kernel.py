from theano import tensor as T
import theano
import pandas as pd
import time

df = pd.read_csv('train.csv')
y = df[['label']]
x = df.drop('label', axis=1).astype('uint32')
[n1, n2] = x.shape
x = x/255

ys = pd.get_dummies(y, columns=['label'], dtype='int')
test_size = 2000

# Kernel Method
start = time.time()
m, m1, m2 = T.matrix(), T.matrix(), T.matrix()
mmT = T.dot(m1, m2.T)
f = theano.function([m1, m2], mmT)
x1, x2 = x[:21000], x[21000:]
K11 = f(x1, x1)
pd.DataFrame(data=K11).to_pickle('K11')
del K11
K12 = f(x1, x2)
pd.DataFrame(data=K12).to_pickle('K12')
del K12
K22 = f(x2, x2)
pd.DataFrame(data=K22).to_pickle('K22')
del K22
end = time.time()
print(end-start, ' seconds')

kernel = T.power(m, 3) + T.power(m, 2) + m + 1
f1 = theano.function([m], kernel)
start = time.time()
K11 = pd.read_pickle('K11')
Kernel11 = f1(K11)
pd.DataFrame(data=Kernel11).to_pickle('Kernel11')
del K11, Kernel11
K12 = pd.read_pickle('K12')
Kernel12 = f1(K12)
pd.DataFrame(data=Kernel12).to_pickle('Kernel12')
del K12, Kernel12
K22 = pd.read_pickle('K22')
Kernel22 = f1(K22)
pd.DataFrame(data=Kernel22).to_pickle('Kernel22')
del K22, Kernel22
end = time.time()
print(end-start, ' seconds')