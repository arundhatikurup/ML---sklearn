from  matplotlib  import pyplot as plt

#import matplotlib.pyplot

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
#print(digits.data)
#print(digits.target)
#print(digits.images[0])

#plt.pyplot.ion()



clf = svm.SVC(gamma=0.001, C=100)
X,y = digits.data[:-10], digits.target[:-10]


clf.fit(X,y)

#X.reshape(-1,1)
print(clf.predict(digits.data[-5]))

plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

