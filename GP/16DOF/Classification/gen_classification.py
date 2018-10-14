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

u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 30)

n_clusters = 2
classifier = cluster.KMeans(n_clusters = n_clusters)

colors = plt.cm.get_cmap('tab10')(np.linspace(0,0.6,n_clusters))

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

    fig = plt.figure()
    ax = Axes3D(fig)
    cnt = 0
    for i in range(n_clusters):
        members = y[g_cluster_labels == i]
        members_u = set(members)
        n_members = len(members)
        n_members_u = len(members_u)
        mean = np.mean(x[g_cluster_labels == i, :], axis = 0)
        var = 2*np.sqrt(np.var(x[g_cluster_labels == i, :], axis = 0))
        # print(x[g_cluster_labels == i, :])
        print('Group', i, 'Members', members_u, 'Unique Count', n_members_u)
        print('\tMean', mean)
        print('\tVar', var)
        # input('')
        # print(x[g_cluster_labels==i, 0], x[g_cluster_labels==i, 1])

        x_plot = mean[0] + var[0]*np.outer(np.sin(u), np.sin(v))
        y_plot = mean[1] + var[1]*np.outer(np.sin(u), np.cos(v))
        z_plot = mean[2] + var[2]*np.outer(np.cos(u), np.ones_like(v))
        ax.plot_surface(x_plot, y_plot, z_plot, alpha = .2, color = colors[i])
        ax.scatter(mean[0], mean[1], mean[2], color = colors[i])
        ax.scatter(x[g_cluster_labels==i,0], x[g_cluster_labels==i,1], x[g_cluster_labels==i,2], label = i, color = colors[i], s = 1)#, facecolors = 'none')
        cnt+= n_members_u

    plt.title(trial)
    ax.legend()
    ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('equal')
    # plt.show()
    print('Total grasps', cnt)
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
