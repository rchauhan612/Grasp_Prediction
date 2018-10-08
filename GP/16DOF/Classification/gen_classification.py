# -*- coding: utf-8 -*-
"""
Ravi's graps prediction
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
from sklearn import decomposition
import os
import itertools
from sklearn import cluster

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.1f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()



# classifier = KNeighborsClassifier(n_neighbors=5)
n_clusters = 8
classifier = cluster.KMeans(n_clusters = n_clusters)

path = '../Trajectories/final_configs'

trials = [i for i in os.listdir(path) if i.split('_')[3].split('.')[0] == 'pca'] # get only the PCA trials

plt.ion()
for trial in trials:
    plt.close('all')
    data = np.load(path + '/' + trial)
    y = data[:, 0]
    x = data[:, 1:]

    classifier.fit(x)
    g_cluster_labels = classifier.labels_

    # print((g_cluster_labels))

    plt.figure(projection = '3d')
    for i in range(n_clusters):
        members = y[g_cluster_labels == i]
        n_members = len(members)
        # print(g_cluster_labels)
        print(x[g_cluster_labels==i, 0], x[g_cluster_labels==i, 1])
        plt.scatter(x[g_cluster_labels==i,0], x[g_cluster_labels==i,1], x[g_cluster_labels==i,2])#, facecolors = 'none')
        # plt.scatter([i]*n_members, members)

    plt.show()
    input(' ')
    plt.close('all')

    # X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.25, random_state=0)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # # classes = classifier.kneighbors_graph(X_train)
    # # # print(classes)
    # # classes = classes.toarray()
    # # for i in range(len(classes)):
    # #     print(y_train[i], y_train[classes[i, :] == 1])
    #     # print(classes[i, :])
    #
    # # Compute confusion matrix
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # np.set_printoptions(precision=1)
    #
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=np.arange(1, 34), normalize=True,
    #                       title='Normalized confusion matrix')
    #
    # plt.show()

    # input(' ')
