import numpy as np

class Rat(object):
    def __init__(self,
                 ratID,
                 alpha = 0.03,
                 beta = 0.1,
                 gamma = 1):

        self.ID = ratID
        #initialize Q value states randomly
        self.Q = [(w - 0.5) / 5 for w in np.random.random(2)]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def softmax(self, array, temp = 0.5):
        tmp = [np.exp(val/temp) for val in array]
        return tmp / np.sum(tmp)

    def make_decision(self):
        p = self.softmax(self.Q, temp = self.beta)
        return np.digitize(np.random.random(), np.cumsum(p))

    def update_beliefs(self, reward, choice):
        self.Q[choice] += self.alpha * (reward - self.Q[choice])

    def rest_of_day(self):
        self.Q = [w * 0.2 for w in rat.Q]
