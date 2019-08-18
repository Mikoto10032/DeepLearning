"""
    @ jetou
    @ support_vector_machine dual algorithm
    @ date 2017 11 12

"""


import numpy as np
from sympy import *


class supprot_vector_machine:
    def __init__(self, feature, label):
        self.feature = np.array(feature)
        self.label =  np.array(label)
        self.N = len(self.feature)
        self.n = -1
        self.alpha = symbols(["alpha" + str(i) for i in range(1, self.N + 1)])
        self.solution = []

    #train
    def transvection(self, a,b): # compute inner product
        result = 0 # conpute inner product function most used gram marix
        offset = len(a)
        for i in range(offset):
            result += a[i] * b[i]
        return result

    def eval_function(self): # compute Master equation
        c = 0
        s = 0 #Temporary variable
        offset = self.N
        for i in range(offset):
            s += self.alpha[i]
            for j in range(offset):
                c += self.alpha[i] * self.alpha[j] * self.label[i][0] * self.label[j][0] * self.transvection(self.feature[i], self.feature[j])
        self.equation = c/2 - s

    def replace_x(self): # add the constraints use alpha * y  and subject to [alpha1 + alpha2 + ... alphaN] = 0
        self.n+=1
        alpha_ = self.alpha[:]
        for i in range(len(alpha_)):
            alpha_[i] = alpha_[i] * self.label[i][0]
        equation = Eq(sum(alpha_),0)
        self.solve_equation =  solve([equation], self.alpha[0])
        self.equation = self.equation.replace(self.solve_equation.keys()[0],
                                              self.solve_equation.get(self.solve_equation.keys()[0]))

    def derivative(self): # conpute derivative
        self.surplus_alpha = self.alpha[:]
        self.equation_list = []
        self.surplus_alpha.remove(self.solve_equation.keys()[0])
        for i in self.surplus_alpha:
            self.equation_list.append(diff(self.equation, i))
        self.solution = solve(self.equation_list)

    def boundary(self): # If the constraint conditions are violated
        min = 10000
        self.surplus_alpha.reverse()

        s = solve(self.replace_model(self.equation_list[:]))

        finaly = []
        for i in range(len(s)):
            m = self.equation
            for j in range(len(s)):
                if j!=i:
                    m = m.replace(symbols("alpha"+str(j+2)), 0)
            finaly.append(m)

        for i in range(len(s)):
            K = Eq(symbols("pp"), finaly[i])
            sad = solve([K, Eq(symbols("alpha" + str(i+2)), s.get(symbols("alpha" + str(i+2))))])[0].get(symbols("pp"))
            if min > sad:
                result = {}
                min = sad
                result[symbols("alpha" + str(i+2))] = s.get(symbols("alpha" + str(i+2)))

        for i in range(2, len(s)+2):
            if symbols("alpha"+str(i)) not in result.keys():
                result[symbols("alpha"+str(i))] = 0

        self.solution = result

    # auxiliary function
    def replace_model(self, lists):
        bound = lists
        k=len(bound)
        for i in range(k):
            for j in range(k):
                if j!=i:
                    bound[i] = bound[i].replace(symbols("alpha"+str(j+2)), 0)
        return bound

    def get_origin(self, source):
        value_list = [Eq(i, source.get(i)) for i in source.keys()]
        value_list.append(Eq(self.solve_equation.keys()[0], self.solve_equation.get(self.solve_equation.keys()[0])))
        return solve(value_list)

    #enter
    def fit(self):
        self.eval_function()
        self.replace_x()
        self.derivative()
        for i in self.solution:
            if (self.solution.get(i)) < 0:
                self.boundary()
                break
        if self.solution == []:
            self.boundary()
        self.solution = self.get_origin(self.solution)

        final_alpha = [self.solution.get(self.alpha[i]) for i in range(len(self.solution))]
        final_alpha = [[i] for i in final_alpha]


        #self.w
        self.w = sum(final_alpha *self.feature*self.label)

        #self.b
        for i in range(len(final_alpha)): #The presence of subscript j makes alpha_j>0
            if final_alpha[i] != 0:
                alpha_j = i
                break
        kk = []
        for i in range(self.N):
            kk.append(self.transvection(self.feature[i], self.feature[alpha_j]))

        self.b = [final_alpha[i] * self.label[i] * kk[i] for i in range(self.N)]
        self.b = self.label[alpha_j] - sum(self.b)

    #prediction
    def prediction(self, feature):
        Mat = np.array(feature).transpose()
        col = Mat.shape[1]

        output = []
        for i in range(col):
            if sum(Mat[:,i] * self.w) + self.b > 0:
                output.append(+1)
            else:
                output.append(-1)

        return output











