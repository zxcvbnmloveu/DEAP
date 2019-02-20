import threading
import time

import numpy
import json
import pickle
from sklearn import model_selection
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import _thread

class SequentialBackwardSelection():
    """
    Feature selection using Sequential Backward Selection
    Parameter
    -----------
    estimator: a classifier or regressor that will be used to do
                feature selection.
    num_of_features: number of features for the feature selection
                    result
    scoring: the scoring method that will be used for selecting
                the best set of features
    test_size: test size ratio
    random_state: random seed value used for train_test_split
    """
    def __init__(self, estimator=KNeighborsClassifier(),
                 num_of_features=5, scoring='f1_score', test_size=0.25,
                 random_state=1):

        self.scoring = scoring
        self.estimator = clone(estimator)
        self.num_of_features = num_of_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fitting the estimator to the data, and finding the best set of
        features from the features of the data. Amount of features in the
        set will be equal to the num_of_features defined
        Parameter
        -----------
        X: Training Vectors, shape=[number_of_samples, number_of_features]
        Y: Target Values, shape=[number_of_samples]
        Attribute
        -----------
        best: integer, index of the feature set that scores the highest
                with the scoring method selected
        indices: best feature set from the combinations of feature set tested
        best_score = score from the test run on the estimator using the
                        best feature set
        scores: list, containing the score from the test run on the estimator
                using the feature sets tested in the method
        """

        total_best_score =[]
        total_best_indice =[]

        iter_subset = numpy.array([i for i in range(X.shape[1])])
        r_num = X.shape[1]
        dict = {}
        while(r_num>self.num_of_features):
            iter_all_score = []
            iter_all_subset = []
            for feature_combination in combinations(iter_subset,r = r_num):
                # print("iter: " + str(feature_combination))
                score = self.calc_score(X, y, feature_combination)
                # print("score: " + str(score))
                iter_all_score.append(score)
                iter_all_subset.append(feature_combination)
            best = np.argmax(iter_all_score)
            total_best_indice.append(iter_all_subset[best])
            total_best_score.append(iter_all_score[best])
            print("iter: " + str(r_num) + " iter_all_subset[best]: " + str(iter_all_subset[best])+" score: " + str(iter_all_score[best]))
            DictData = (str(iter_all_subset[best]),str(iter_all_score[best]))
            dict[str(r_num)] = DictData
            iter_subset =  numpy.array(iter_all_subset[best])
            r_num = r_num - 1

        best = np.argmax(total_best_score)
        self.indices = total_best_indice[best]
        self.best_score = total_best_score[best]
        print("best indices: " + str(self.indices))
        print("best score: " + str(self.best_score))
        # return self
        return dict

        # self.scores = []
        # self.subsets = []
        # for feature_combination in combinations(range(X[:, 2:10].shape[1]),
        #                                         r=self.num_of_features):
        #     print(feature_combination)
        #     score = self.calc_score(X, y, feature_combination)
        #     self.scores.append(score)
        #     self.subsets.append(feature_combination)
        #
        # best = np.argmax(self.scores)
        # self.indices = self.subsets[best]
        # self.best_score = self.scores[best]
        # print(self.indices)
        # print(self.best_score)
        # return self

    def transform(self, X):
        """
        Transform a data vector into a new data vector containing only the
        features from the best set of features obtained from the fit function
        Parameter
        -----------
        X: Data Vectors, shape=[number_of_samples, number_of_features]
        Data with the same features as the ones used for fitting
        Attribute
        -----------
        indices: best feature set from the combinations of feature set tested
        """
        return X[:, self.indices]

    def calc_score(self, X, y, indices):
        """
        Scoring an estimator using data with only a specific set of features
        Parameter
        -----------
        X: Training Vectors, shape=[number_of_samples, number_of_features]
        Y: Target Values, shape=[number_of_samples]
        indices: column index, features that will be used from X
        """
        #交叉验证法
        # X_train, X_test, \
        #     y_train, y_test = train_test_split(X, y, test_size=self.test_size,
        #                                        random_state=self.random_state)
        # self.estimator.fit(X_train[:, indices], y_train)
        # y_prediction = self.estimator.predict(X_test[:, indices])
        # score = self.scoring(y_test, y_prediction)
        #留一法
        loo=LeaveOneOut()
        result = model_selection.cross_val_score(self.estimator, X[:, indices], y, cv=loo, scoring=self.scoring)
        score = result.mean()

        return score

class myThread (threading.Thread):
    def __init__(self, threadID, name, X, y):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.X = X
        self.y = y
    def run(self):
        print ("开始线程：" + self.name)
        s = SequentialBackwardSelection(estimator = KNeighborsClassifier(), scoring = 'accuracy', num_of_features=1)
        dict = s.fit(self.X,self.y)
        # 写入 JSON 数据
        with open("E:/DataSet/data/SBS/" + str(self.name)+ ".json", 'w') as f:
            json.dump(obj = dict, fp = f)

        print ("退出线程：" + self.name)


if __name__ == "__main__":

    # # 读取数据
    # with open('data.json', 'r') as f:
    #     data = json.load(f)


    #noise_valence
    file_x = 'E:/DataSet/data/features_noise.dat'
    file_y = 'E:/DataSet/data/label_class_0.dat'
    X1 = numpy.genfromtxt(file_x, delimiter=' ')
    y1 = numpy.genfromtxt(file_y, delimiter=' ')
    #clear_valence
    file_x = 'E:/DataSet/data/features_clear.dat'
    file_y = 'E:/DataSet/data/label_class_0.dat'
    X2 = numpy.genfromtxt(file_x, delimiter=' ')
    y2 = numpy.genfromtxt(file_y, delimiter=' ')
    # #noise_arousal
    file_x = 'E:/DataSet/data/features_noise.dat'
    file_y = 'E:/DataSet/data/label_class_1.dat'
    X3 = numpy.genfromtxt(file_x, delimiter=' ')
    y3 = numpy.genfromtxt(file_y, delimiter=' ')
    # #clear_arousal
    file_x = 'E:/DataSet/data/features_clear.dat'
    file_y = 'E:/DataSet/data/label_class_1.dat'
    X4 = numpy.genfromtxt(file_x, delimiter=' ')
    y4 = numpy.genfromtxt(file_y, delimiter=' ')

    # estimator=LogisticRegression(solver='liblinear')
    # estimator=SVC(gamma = 'scale')
    # estimator=KNeighborsClassifier()
    # estimator=DecisionTreeClassifier()

    file_x = "E:/DataSet/data/features_clear_less.dat"
    file_y = 'E:/DataSet/data/label_class_0.dat'
    X5 = numpy.genfromtxt(file_x, delimiter=' ')
    y5 = numpy.genfromtxt(file_y, delimiter=' ')

    file_x = "E:/DataSet/data/features_clear_less.dat"
    file_y = 'E:/DataSet/data/label_class_1.dat'
    X6 = numpy.genfromtxt(file_x, delimiter=' ')
    y6 = numpy.genfromtxt(file_y, delimiter=' ')

    # 创建新线程
    # thread1 = myThread(1, "noise_valence", X1,y1)
    thread2 = myThread(2, "clear_valence_18fea", X5,y5)
    # thread3 = myThread(3, "noise_arousal", X3,y3)
    thread4 = myThread(4, "clear_arousal_18fea", X6,y6)
    # 开启新线程
    # thread1.start()
    thread2.start()
    # thread3.start()
    thread4.start()
    # thread1.join()
    thread2.join()
    # thread3.join()
    thread4.join()
    print ("退出主线程")
