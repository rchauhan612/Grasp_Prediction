import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # # print(plt.xticks())
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, format(cm[i, j], fmt),
        #          horizontalalignment="center",
        #          color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



reg_res = np.load('reg_res.npy')
gp_res = np.load('gp_res.npy')[:, [0, 6]]
# Compute confusion matrix
cnf_matrix_res = confusion_matrix(reg_res[:, 0], reg_res[:, 1])
cnf_matrix_gp = confusion_matrix(gp_res[:, 0], gp_res[:, 1])


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_res, classes=str(np.arange(1, 34)), normalize=True,
                      title='reg confusion matrix')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_gp, classes=str(np.arange(1, 34)), normalize=True,
                      title='gp confusion matrix')

plt.show()
