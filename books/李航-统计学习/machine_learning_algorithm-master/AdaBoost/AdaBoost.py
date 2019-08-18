"""
    @ jetou
    @ AdaBoost algorithm
    @ date 2017 11 19

"""
from weaker_classifier import *
import math

class adaBoost:
    def __init__(self, feature, label, Epsilon = 0):
        self.feature = np.array(feature)
        self.label   = np.array(label)
        self.Epsilon = Epsilon
        self.N       = self.feature.shape[0]
        self.error   = 1
        self.count_error = 1

        self.alpha = []
        self.classifier = []
        self.W = [1.0 / self.N for i in range(self.N)]


    def sign(self, value):
        if value > 0:
            value = 1
        elif value < 0:
            value = -1
        else:
            value = 0

        return value

    def update_W_(self):
        update_W = []
        z = 0
        for i in range(self.N):
            pe = np.array([self.feature[i]])
            z += self.W[i] * math.exp(-1 * self.alpha[-1] * self.label[i] * self.classifier[-1].prediction(pe))

        for i in range(self.N):
            kk = np.array([self.feature[i]])
            w = self.W[i] * math.exp(-1 * self.alpha[-1] * self.label[i] * self.classifier[-1].prediction(kk)) / z
            update_W.append(w)
        self.W = update_W

    def __alpha__(self):
       self.alpha.append(math.log((1-self.error)/self.error)/2)


    def prediction(self, label):
        finaly_prediction = []
        classifier_offset = len(self.classifier)

        for i in range(len(label)):
            result = 0
            for j in range(classifier_offset):
                pe = np.array([label[i]])
                result += self.alpha[j] * self.classifier[j].prediction(pe)
            finaly_prediction.append(self.sign(result))

        return finaly_prediction

    def complute_error(self):
        # compute error
        result = self.prediction(self.feature)
        count_error = 0
        for i in range(self.N):
            if result[i] * self.label[i] < 0:
                count_error+=1
        self.count_error = count_error / (self.N * 1.0)  #compute error%



    def train(self):
        while(self.count_error > self.Epsilon):
            classifier = weake_classifier(self.feature, self.label, self.W)
            self.classifier.append(classifier)
            classifier.train()
            self.error, self.W, dem = classifier.get_information()
            self.__alpha__()
            self.update_W_()
            self.complute_error()


