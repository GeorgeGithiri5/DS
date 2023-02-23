from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

iris = load_iris()

X = iris.data
Y = iris.target

clf = GaussianNB()
clf.fit(X, Y)