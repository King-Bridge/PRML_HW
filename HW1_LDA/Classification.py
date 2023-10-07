import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import data
n_samples = 300
cancer = datasets.load_breast_cancer()
X,y = cancer['data'], cancer['target']
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# Example: use LogisticRegression and evaluate the model
logreg = linear_model.LogisticRegression(C=1e3)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('precision: ', accuracy_score(y_test, y_pred))

# TODO: Write your own LDA and evaluate it
class LDA():
    def __init__(self):
        self.mu = [] # 均值
        self.sigma = [] # 协方差矩阵
        self.S_b = None # 类间散度矩阵
        self.S_t = None # 类内散度矩阵
    
    def fit(self, X, y):
        # 将不同类的数据分开
        x = [[],[]]
        for i in range (X.shape[0]):
            x[y[i]].append(X[i])

        # 计算均值
        for i in range (2):
            self.mu.append(np.zeros(X.shape[1]))
            for j in range (len(x[i])):
                self.mu[i] += x[i][j] / len(x[i])
        print("各样本均值为：\n", self.mu)

        # 计算协方差矩阵
        for i in range (2):
            self.sigma.append(np.cov(x[i]))
        print("各样本协方差矩阵为：\n", self.sigma, self.sigma[0].shape, self.sigma[1].shape)

    def predict(self, X):
        pass

lda = LDA()
lda.fit(X_train, y_train)
# y_pred = lda.predict(X_test)
# print('precision: ', accuracy_score(y_test, y_pred))