import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, std, classes,
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
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    # # print(plt.xticks())
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i-.1, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.text(j, i+.1, '(' + format(std[i, j], fmt) + ')',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


subjects = ['Ravi', 'Bijo', 'Dan', 'Shubh']
res = np.array([[0, 0]])
cnf_matrix_all = []
for name in subjects:
    res_temp = np.load(name + '/GPs/glove_pred_results.npy')
    test_temp = [res_temp[i][0] for i in range(len(res_temp))]
    pred_temp = [res_temp[i][1] for i in range(len(res_temp))]
    cnf_matrix_all.append(confusion_matrix(test_temp, pred_temp))
    res = np.vstack((res, res_temp))
# Compute confusion matrix
res = res[1:]
# print(cnf_matrix_all)

test = [res[i][0] for i in range(len(res))]
pred = [res[i][1] for i in range(len(res))]

cnf_matrix_res = confusion_matrix(test, pred)

std = np.zeros(cnf_matrix_res.shape)

# print(cnf_matrix_res.shape)

for i in range(cnf_matrix_res.shape[0]):
    for j in range(cnf_matrix_res.shape[1]):
        std[i, j] = np.std([cnf_matrix_all[k][i, j]/30 for k in range(len(subjects))])
        print(i, j, [cnf_matrix_all[k][i, j]/30 for k in range(len(subjects))])

print(std)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_res, std, classes=['ring', 'tri', 'cyl', 'qd', 'ven'], normalize=True,
                      title='reg confusion matrix')


plt.show()
