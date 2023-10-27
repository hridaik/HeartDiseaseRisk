
#import pickle




#import numpy as np

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')


#import matplotlib.pyplot as plt
#from matplotlib import rcParams


import warnings
warnings.filterwarnings('ignore')




dataset = pd.read_csv('dataset.csv')


"""
rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)

plt.colorbar()
"""


dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
y = dataset['condition']
X = dataset.drop(['condition'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)



"""
knn_classifier = pickle.load('classifier', 'rb')                                                                                                            
"""
knn_scores = []



for k in range(1,11):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

"""
knn_pkl = open('classifier', 'wb')
pickle.dump(knn_classifier, knn_pkl)
"""

for k in range(1,11):
    knn_scores.append(knn_classifier.score(X_test, y_test))

for i in range(len(knn_scores)):
    knn_scores[i] = round(knn_scores[i], 5)*100

print(knn_scores)

"""
plt.plot([k for k in range(1, 11)], knn_scores, color = 'blue')
for i in range(1,11):
    if i == 5:
        plt.text(5, knn_scores[4], 'Optimal (84%, 5)')
    plt.text(i, knn_scores[i-1], '')
plt.xticks([i for i in range(1, 12)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy (%)')
plt.title('K Neighbors Classifier Accuracy For Different Values of K')
"""



