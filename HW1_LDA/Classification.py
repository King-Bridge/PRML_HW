import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain

# import data
n_samples = 300
cancer = datasets.load_breast_cancer()
X,y = cancer['data'], cancer['target']
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# # Example: use LogisticRegression and evaluate the model
# logreg = linear_model.LogisticRegression(C=1e3)
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# print('precision: ', accuracy_score(y_test, y_pred))

def draw_histogram(x_min, x_max, x_data, w, color_list):
    x_axis1 = [x for x in range (x_min, x_max, 1)]
    x_axis2 = [x+1 for x in range (x_min, x_max, 1)]
    x_axis = list(chain.from_iterable(zip(x_axis1, x_axis2)))
    labels = ['negative count ratio', 'positive count ratio']
    for i in range (2):
        points = []
        for sam in x_data[i]:
            points.append(float(np.dot(sam, w)))
        # print(points)
        s = pd.cut(points, bins=[x for x in range (-15, 15, 1)])
        # print(s.value_counts())
        values = (s.value_counts()).values
        values = [value/len(x_data[i]) for value in values]
        # print(values) 
        y_axis = [value for value in values for i in range (2)]
        # print(y_axis)
        # plt.step(x_axis, y_axis)
        if i == 0: color = color_list[0]
        else:color = color_list[1]
        plt.plot(x_axis, y_axis, 'o--', color=color, alpha=0, linewidth=5)
        plt.fill_between(x_axis, y_axis, 0, color=color, alpha=0.5, label=labels[i])


# TODO: Write your own LDA and evaluate it
class LDA():
    def __init__(self):
        self.mean = [] # 投影后的均值
        self.var = [] # 投影后的方差
        self.w = None # 投影线

    def draw_gauss(self):
        x = np.arange(-12, 20, 0.1)
        y = []
        for i in range (2):
            y.append((1 / (np.sqrt(2*np.pi) * self.var[i])) * np.e ** ( - (x - self.mean[i]) ** 2 / (2 * self.var[i] ** 2)))
        plt.plot(x, y[0], 'b-', linewidth=2, label='negative (fit by normal distribution)')
        plt.plot(x, y[1], 'r-', linewidth=2, label='positive (fit by normal distribution)')
        plt.legend(loc='upper right')

    def visualize(self, X_train, X_test, y_train, y_test):
        
        plt.figure(figsize=(24, 6))
        plt.subplot(121)
        plt.title('train data')
        # 先画实际分布图像
        x_line = [[],[]]
        for i in range (X_train.shape[0]):
            x_line[y_train[i]].append(X_train[i])
        draw_histogram(-15, 14, x_line, self.w, ['blue', 'red'])
        self.draw_gauss()

        plt.subplot(122)
        plt.title('test data')
        # 最后画预测图像
        x_points = [[],[]]
        for i in range (X_test.shape[0]):
            x_points[y_test[i]].append(X_test[i])
        draw_histogram(-15, 14, x_points, self.w, ['purple', 'orange'])
        self.draw_gauss()

        plt.show()
        


    def fit(self, X, y):
        # 将不同类的样本分开
        x = [[],[]]
        for i in range (X.shape[0]):
            x[y[i]].append(X[i])

        # 计算均值
        Mu = []
        for i in range (2):
            Mu.append(np.zeros(X.shape[1]))
            for j in range (len(x[i])):
                Mu[i] += x[i][j] / len(x[i])
        # print("各样本均值为：\n", Mu)

        # 计算协方差矩阵
        Sigma = []
        for i in range (2):
            Sigma.append(np.cov(x[i], rowvar=False))
        # print("各样本协方差矩阵为：\n", Sigma, Sigma[0].shape, Sigma[1].shape)

        # 计算类内、类间散度矩阵
        mu_delta = np.array([Mu[0] - Mu[1]])
        Sb = np.dot(np.transpose(mu_delta), mu_delta)
        Sw = Sigma[0] + Sigma[1]
        # print(Sb.shape, Sw.shape)

        # 计算投影线
        self.w = np.dot(np.linalg.inv(Sw), np.transpose(mu_delta))
        # print(self.w, self.w.shape)

        # 计算样本投影之后的均值和方差
        # for i in range (2):
        #     self.var.append(np.dot(np.transpose(self.w), np.dot(Sigma[i], self.w)))
        #     self.mean.append(np.dot(np.array([Mu[1]]), self.w))
        ### 这里用投影后的列表重新算均值和方差精度更高
        for i in range (2):
            proj = []
            for sam in x[i]:
                proj.append(np.dot(sam, self.w))
            self.mean.append(np.mean(proj))
            self.var.append(np.var(proj))
        print("投影后的均值和方差分别为：\n", self.mean, "\n", self.var)

    def predict(self, X):
        result = []
        for sam in X:
            p = []
            # 比较数据点投影后在两个类里对应的高斯分布概率，取其高
            for i in range (2):
                proj = np.dot(sam, self.w)
                pi = (1 / (np.sqrt(2*np.pi) * self.var[i])) * np.e ** ( - (proj - self.mean[i]) ** 2 / (2 * self.var[i] ** 2))
                p.append(pi)
            result.append(np.argmax(p))
        return result

lda = LDA()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print('precision: ', accuracy_score(y_test, y_pred))

lda.visualize(X_train, X_test, y_train, y_test)