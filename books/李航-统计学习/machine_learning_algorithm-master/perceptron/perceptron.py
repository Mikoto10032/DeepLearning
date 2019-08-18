"""
    @ jetou
    @ perceptron Duality
    @ date 2017 11 01

"""
import numpy as np

class perceptron:
    def __init__(self, feature, label, gama=1):
        self.feature = feature.transpose()
        self.label = label.transpose()
        self.__feature = feature

        self.row = self.feature.shape[0]
        self.col = self.feature.shape[1]

        self.alpha = [0] * self.col
        self.b = 0
        self.gama = gama

    def gram(self): # compute gram matrix
        self.gram_mat = np.empty(shape=(self.col, self.col))
        for i in range(self.col):
            for j in range(self.col):
                self.gram_mat[i][j] = self.inner(self.__feature[i], self.__feature[j])

    def inner(self, a, b):
        result = a[0] * b[0] + a[1] * b[1]
        return result

    def misinterpreted(self, yi, index):
        pa1 = self.alpha * self.label
        pa2 = self.gram_mat[:,index]
        num = yi * (np.dot(pa1, pa2) + self.b)
        return num

    def fit(self, alpha, b):
        label = self.label.flatten(1)
        for i in range(self.col):
            while self.misinterpreted(label[i],i) <= 0:
                self.alpha[i] += self.gama
                self.b += self.gama * label[i]
        if alpha!=self.alpha or b != self.b: #To prevent the front of a bit not fit
            self.fit(self.alpha, self.b)
        return self.alpha

    def train(self):
        self.gram()
        self.w = sum((self.fit(1,2)*self.feature*self.label).transpose()) #self.fit(x,y) Any two numbers

    def dot_prediction(self, A, B):
        assert len(A) == len(B)
        summer = 0
        for i in range(len(A)):
            summer += A[i] * B[i]

        return summer

    def prediction(self, feature):
        Mat = np.array(feature).transpose()
        col = Mat.shape[1]

        output = []
        for i in range(col):
            if self.dot_prediction(self.w, Mat[:,i]) + self.b > 0:
                output.append(+1)
            else:
                output.append(-1)

        return output

    def get_wandb(self): # plot w b
        return self.w, self.b





