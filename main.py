import pandas as pd
import numpy as np
from scipy.special import expit
from matplotlib import pyplot as plt
import time
from scipy.sparse import csr_matrix
import csv


df = pd.read_csv('train.csv')
y = df[['label']]
x = df.drop('label', axis=1).astype('uint32')
x = x/255

ys = pd.get_dummies(y, columns=['label'], dtype='int')
alpha = 0.01
test_size = 2000


# Stochastic Gradient Ascent
theta = np.zeros(784)
test_error1 = np.zeros(10*(42000-test_size)//800 + 1)
test_error2 = np.zeros(11)
test_error3 = np.zeros(31)
test_error4 = np.zeros(1001)
training_error1 = np.zeros(10*(42000-test_size)//800 + 1)
training_error2 = np.zeros(11)
training_error3 = np.zeros(31)
training_error4 = np.zeros(1001)
xs = [0]

temp = np.dot(theta, x[42000-test_size:].T)
pred = expit(temp)
initial_test_residual = np.linalg.norm(ys.loc[42000-test_size:, 'label_0'] - 1*(pred>=0.5))**2
temp = np.dot(theta, x[:42000-test_size].T)
pred = expit(temp)
initial_train_residual = np.linalg.norm(ys.loc[:42000-test_size-1, 'label_0'] - 1*(pred>=0.5))**2
test_error1[0] = initial_test_residual
test_error2[0] = initial_test_residual
test_error3[0] = initial_test_residual
test_error4[0] = initial_test_residual
training_error1[0] = initial_train_residual
training_error2[0] = initial_train_residual
training_error3[0] = initial_train_residual
training_error4[0] = initial_train_residual

for N in range(10):
    print(N)
    for i in range(42000-test_size):
        temp = np.dot(theta, x.iloc[i])
        pred = expit(temp)
        theta += alpha*(ys.loc[i, 'label_0']-pred)*x.iloc[i]
        if i%800 == 0:
            temp = np.dot(theta, x[42000-test_size:].T)
            pred = expit(temp)
            test_error1[N*(42000-test_size)//800+i//800+1] = np.linalg.norm(ys.loc[42000-test_size:, 'label_0'] -
                                                                              1*(pred>=0.5))**2
            temp = np.dot(theta, x[:42000-test_size].T)
            pred = expit(temp)
            training_error1[N*(42000-test_size)//800+i//800+1] = np.linalg.norm(ys.loc[:42000-test_size-1, 'label_0'] -
                                                                              1*(pred>=0.5))**2
            xs.append(N*(42000-test_size)+i+1)

plt.plot(xs[1:], test_error1[1:]/2000)
plt.plot(xs[1:], training_error1[1:]/40000)
plt.title('Stochastic Gradient Ascent % Misclassification')
plt.xlabel('Stochastic Gradient Ascent Epochs')
plt.xticks(ticks=list(range(40000, 400001, 40000)), labels=[str(elem) for elem in range(1, 11)])
plt.legend(['test', 'train'])
plt.show()


# Batch Gradient Ascent
theta = np.zeros(784)
for N in range(10):
    theta_next = theta
    print(N)
    for i in range(42000-test_size):
        temp = np.dot(theta, x.iloc[i])
        p = expit(temp)
        theta_next += alpha*(ys.loc[i, 'label_0']-p)*x.iloc[i]
    theta = theta_next
    temp = np.dot(theta, x[42000-test_size:].T)
    pred = expit(temp)
    test_error2[N+1] = np.linalg.norm(ys.loc[42000-test_size:, 'label_0'] - 1*(pred>=0.5))**2
    temp = np.dot(theta, x[:42000-test_size].T)
    pred = expit(temp)
    training_error2[N+1] = np.linalg.norm(ys.loc[:42000-test_size-1, 'label_0'] - 1*(pred>=0.5))**2

plt.plot(range(1, 11), test_error2[1:]/2000)
plt.plot(range(1, 11), training_error2[1:]/40000)
plt.title('Batch Gradient Ascent % Misclassification')
plt.xlabel('Batch Gradient Ascent Epochs')
plt.legend(['test', 'train'])
plt.show()


# Kernel Method
gamma = np.zeros(42000-test_size)
alpha = 0.01
for N in range(30):
    print(N)
    gamma_new = gamma
    K11 = pd.read_pickle('Kernel11')
    sum = np.zeros(21000)
    sum += np.dot(K11, gamma[:21000])
    del K11
    K12 = pd.read_pickle('Kernel12')
    sum += np.dot(K12.iloc[:, :19000], gamma[21000:])
    gamma_new[:21000] += alpha*(ys.loc[:20999, 'label_0']-expit(sum))

    sum = np.zeros(21000-test_size)
    sum += np.dot(gamma[:21000], K12.iloc[:, :19000])
    del K12
    K22 = pd.read_pickle('Kernel22')
    sum += np.dot(gamma[21000:], K22.iloc[:19000, :19000])
    del K22
    gamma_new[21000:] += alpha*(ys.loc[21000:42000-test_size-1, 'label_0']-expit(sum))
    gamma = gamma_new

    K11 = pd.read_pickle('Kernel11')
    temp = np.dot(gamma[:21000], K11)
    del K11
    K12 = pd.read_pickle('Kernel12')
    temp += np.dot(gamma[21000:], K12.iloc[:, :19000].T)
    pred1 = expit(temp)
    temp = np.dot(gamma[:21000], K12.iloc[:, :19000])
    del K12
    K22 = pd.read_pickle('Kernel22')
    temp += np.dot(gamma[21000:], K22.iloc[:19000, :19000])
    del K22
    pred2 = expit(temp)
    pred = np.hstack((pred1, pred2))
    training_error3[N+1] = np.linalg.norm(ys.loc[:42000-test_size-1, 'label_0'] - 1 * (pred >= 0.5)) ** 2
    print(training_error3[N+1])

    K12 = pd.read_pickle('Kernel12')
    temp = np.dot(gamma[:21000], K12.iloc[:, 19000:])
    del K12
    K22 = pd.read_pickle('Kernel22')
    temp += np.dot(gamma[21000:], K22.iloc[:19000, 19000:])
    del K22
    pred = expit(temp)
    test_error3[N+1] = np.linalg.norm(ys.loc[42000-test_size:, 'label_0'] - 1 * (pred >= 0.5)) ** 2
    print(test_error3[N+1])

plt.plot(test_error3[1:]/2000)
plt.plot(training_error3[1:]/40000)
plt.title('Kernel Method % Misclassification')
plt.xlabel('Epochs')
plt.legend(['test', 'train'])
plt.show()

"""
wrong = ys.loc[:42000-test_size-1].index[ys.loc[:42000-test_size-1, 'label_0'] != 1*(pred >= 0.5)].tolist()
plt.figure(figsize=(10, 8))
for i, idx in enumerate(wrong[:16]):
    plt.subplot(4, 4, i + 1)
    plt.suptitle('Random images from the train dataset', fontsize=18)
    img = df.drop(['label'], axis=1).iloc[idx].values
    img = img.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {df.loc[idx, 'label']}")
    plt.axis('off')

plt.tight_layout()
plt.show()
"""

output = np.zeros((10, 784))
k_fold = 10
alphas = [1]
prev = -float('inf')
reg = 0
for _ in range(k_fold):
    shuffle_index = np.random.permutation(42000)
    x_new, y_new, ys_new = x.iloc[shuffle_index], y.iloc[shuffle_index], ys.iloc[shuffle_index]
    theta = np.zeros((10, 784))
    ys_new = ys_new.iloc[:40000].to_numpy()
    #objectives = np.zeros(1000)
    for N in range(100):
        print(N)
        temp = np.matmul(theta, x_new.iloc[:40000].T)
        # Subtract a constant before applying softmax function to prevent overflow
        temp = np.exp(temp.sub(temp.max(axis=0), axis=1))
        temp = temp.div(temp.sum(axis=0), axis=1)
        labels = ['label_' + str(k) for k in range(10)]
        """
        line_search = []
        for alpha in alphas:
            theta_next = theta.copy()
            bracket = ys.T - temp
            temp2 = np.matmul(bracket, x.iloc[:40000])
            theta_next += alpha*temp2/40000 - alpha*reg*theta_next
            temp3 = np.matmul(theta_next, x.T)
            temp3 = np.exp(temp3.sub(temp3.max(axis=0), axis=1))
            temp3 = temp3.div(temp3.sum(axis=0), axis=1)
            objective = [temp3.iloc[y.loc[i, 'label'], i] for i in range(40000)]
            objective = [elem if elem > 1e-20 else 1e-20 for elem in objective]
            objective = (np.log(objective).sum())/40000
            print('Alpha: ', alpha, ', Objective: ', objective)
            line_search.append(objective)
            if len(line_search) >= 2 and objective < line_search[-2]:
                break
        if len(line_search) == 0 or max(line_search) <= prev:
            print('STOP')
            break
        alpha = alphas[line_search.index(max(line_search))]
        print('Alpha: ', alpha)
        """
        alpha = alphas[0]
        bracket = ys_new.T - temp
        temp2 = np.matmul(bracket, x_new.iloc[:40000])
        theta += alpha * temp2/40000 - alpha*reg*theta
        """
        temp3 = np.matmul(theta, x.T)
        temp3 = np.exp(temp3.sub(temp3.max(axis=0), axis=1))
        temp3 = temp3.div(temp3.sum(axis=0), axis=1)
        pred = temp3.idxmax(axis=0)
        print(sum(1*(y.loc[:39999, 'label'] != pred[:40000])), sum(1*(y.loc[40000:, 'label'] != pred[40000:])))
        """
        """
        objective = [temp3.iloc[y.loc[i, 'label'], i] for i in range(40000)]
        objective = [elem if elem > 1e-20 else 1e-20 for elem in objective]
        objective = (np.log(objective).sum())/40000
        print('Objective: ', objective)
        prev = objective
        """
    temp3 = np.matmul(theta, x_new.T)
    temp3 = np.exp(temp3.sub(temp3.max(axis=0), axis=1))
    temp3 = temp3.div(temp3.sum(axis=0), axis=1)
    pred = temp3.idxmax(axis=0)
    print(sum(1 * (y_new.iloc[:40000]['label'] != pred[:40000])), sum(1 * (y_new.iloc[40000:]['label'] != pred[40000:])))
    output += theta
output /= k_fold
temp3 = np.matmul(output, x.T)
temp3 = np.exp(temp3.sub(temp3.max(axis=0), axis=1))
temp3 = temp3.div(temp3.sum(axis=0), axis=1)
pred = temp3.idxmax(axis=0)
print(sum(1 * (y.loc[:39999, 'label'] != pred[:40000])), sum(1 * (y.loc[40000:, 'label'] != pred[40000:])))

df = pd.DataFrame(data=output)
df.to_pickle('theta3')

theta = pd.read_pickle('theta3').to_numpy()
test = pd.read_csv('test.csv')
temp = np.matmul(theta, test.T)
pred = temp.idxmax(axis=0)
output = pd.DataFrame(data={'ImageId': test.index+1, 'Label': pred})
output.set_index('ImageId', inplace=True)
output.to_csv('output.csv')