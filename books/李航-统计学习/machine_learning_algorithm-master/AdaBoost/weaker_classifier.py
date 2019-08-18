"""
    @ jetou
    @ weaker_classifier algorithm
    @ date 2017 11 19

"""
import numpy as np
class weake_classifier:
    def __init__(self, feature, label, W = None):
        self.feature = np.array(feature)
        self.label   = np.array(label)

        self.setlable = np.unique(label)
        self.feature_dem = self.feature.shape[1]
        self.N = self.feature.shape[0]

        if W != None:
            self.W = np.array(W)
        else:
            self.W = [1.0 / self.N for i in range(self.N)]


    def prediction(self, feature):
        test_feature = np.array(feature)
        output = np.ones((test_feature.shape[0],1))
        output[test_feature[:, self.demention] * self.finaly_label < self.threshold * self.finaly_label] = -1


        return output

    def __str__(self):
        string  = "opt_threshold:" + str(self.threshold)    + "\n"
        string += "opt_demention:" + str(self.demention)    + "\n"
        string += "opt_errorRate:" + str(self.error)        + "\n"
        string += "opt_label    :" + str(self.finaly_label) + "\n"
        string += "weights      :" + str(self.W)            + "\n"

        return string

    def best_along_dem(self, demention, label):
        feature_max = np.max(self.feature)
        feature_min = np.min(self.feature)
        step = (feature_max - feature_min) / (self.N * 1.0)
        min_error = self.N * 1.0

        for i in np.arange(feature_min, feature_max, step):
            output = np.ones((self.N, 1))
            output[self.feature[:, demention] * label < i * label] = -1

            errorRate = 0.0
            for j in range(self.N):
                if output[j] * self.label[j] < 0:
                    errorRate += self.W[j]

            if errorRate < min_error:
                min_error = errorRate
                threshold = i

        return  threshold, min_error

    def train(self):
        self.error = self.N * 1.0
        for demention in range(self.feature_dem):
            for label in self.label:
                threshold, err = self.best_along_dem(demention, label)
                if self.error > err:
                    self.error = err
                    self.finaly_label = label
                    self.threshold = threshold
                    self.demention = demention

    def get_information(self):
        return self.error, self.W, self.demention
