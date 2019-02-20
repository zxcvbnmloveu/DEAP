import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import time

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def cross_validate0(file_x, file_y):
    
    # Get Data
    # file_x = 'E:/DataSet/data/features_sampled.dat'
    # file_x = 'E:/DataSet/data/features_raw.dat'
    # file_x = 'E:/DataSet/data/features_noise.dat'
    # file_x = 'E:/DataSet/data/features_clear.dat'

    # file_y = 'E:/DataSet/data/label_class_0.dat'
    # file_y = "E:\DataSet\data\label_class_0_3class.dat"
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    print(X.shape)
    X = StandardScaler().fit_transform(X)

    # permutation = numpy.random.permutation(y.shape[0])
    # X = X[permutation,:]
    # print(y.shape)

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear')))
    models.append(('SVC', SVC()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DT', DecisionTreeClassifier()))
    # models.append(('RF', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456, criterion='entropy')))
    scoring = 'accuracy'

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    numpy.random.seed(10)
    shuffle_indices = numpy.random.permutation(numpy.arange(len(y)))
    x_shuffled = X[shuffle_indices]  # 将文本和标签打乱
    y_shuffled = y[shuffle_indices]

    # Cross Validate
    results = []
    names = []
    timer = []
    print('Model | Mean of CV | Std. Dev. of CV | Time')
    for name, model in models:
        start_time = time.time()
        kfold = model_selection.KFold(n_splits=10, random_state=42)
        cv_results = model_selection.cross_val_score(model, x_shuffled, y_shuffled, cv=kfold, scoring=scoring)
        t = (time.time() - start_time)
        timer.append(t)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f) %f s" % (name, cv_results.mean(), cv_results.std(), t)
        print(msg)

    # #留一法验证
    # loo=LeaveOneOut()
    # for name, model in models:
    #     start_time = time.time()
    #     # [:,(25, 90, 91)]
    #     cv_results = model_selection.cross_val_score(model, x_shuffled, y_shuffled, cv=loo, scoring=scoring)
    #     names.append(name)
    #     t = (time.time() - start_time)
    #     msg = "%s: %f (%f) %f s" % (name, cv_results.mean(), cv_results.std(), t)
    #     print(msg)

if __name__ == '__main__':
    print("抽样2class:")
    cross_validate0(file_x = "E:/DataSet/data/features_22680.dat",file_y = 'E:/DataSet/data/label_class_0.dat')
    print()
    print("抽样3class:")
    cross_validate0(file_x = "E:/DataSet/data/features_22680.dat",file_y = 'E:/DataSet/data/label_class_0_3class.dat')
    print()

# 1000MA less 2Classes:
# Noise:
# (1280, 18)
# LR: 0.637500 (0.480722) 5.008615 s
# SVC: 0.637500 (0.480722) 51.621563 s
# KNN: 0.590625 (0.491719) 2.271543 s
# DT: 0.637500 (0.480722) 2.020986 s
#
# 2Classes:
# Noise:
# (1280, 18)
# LR: 0.592187 (0.491428) 4.168523 s
# SVC: 0.544531 (0.498013) 50.948932 s
# KNN: 0.614062 (0.486816) 2.267431 s
# DT: 0.592187 (0.491428) 1.989042 s
