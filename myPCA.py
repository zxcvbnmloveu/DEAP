# Get Data
from sklearn.decomposition import PCA
import numpy
from sklearn.preprocessing import StandardScaler

file_x = 'E:/DataSet/data/features_clear.dat'

file_y = 'E:/DataSet/data/label_class_0.dat'

X = numpy.genfromtxt(file_x, delimiter=' ')
y = numpy.genfromtxt(file_y, delimiter=' ')

# X = StandardScaler().fit_transform(X)

print(X.shape)

pca=PCA(n_components=20,svd_solver='full')
pca.fit(X[:,0:99])
newData = pca.transform(X[:,0:99])
print(newData.shape)
