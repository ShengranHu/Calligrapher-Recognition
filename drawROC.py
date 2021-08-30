import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

import torch
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

from data_process.calligraphy import get_data_transforms, calligraphy
from inference import load_model

device = 'cpu'

n_classes = 5

plt.rcParams['font.sans-serif'] = ['SimHei']  # chinese label support
plt.rcParams['axes.unicode_minus'] = False  # unicode_minus character support

idx_to_class = {0: u'张旭',
                1: u'褚遂良',
                2: u'赵孟頫',
                3: u'钟绍京',
                4: u'颜真卿'}

if __name__ == '__main__':

    ckpt_path = '/store_point/weights-res101-balanced-97.19.pt'

    net = load_model(
        ckpt_path=ckpt_path)

    train_transform, valid_transform = get_data_transforms()

    train_data = calligraphy().get_train_data(
        transform=train_transform)
    valid_data = calligraphy().get_val_data(
        transform=valid_transform)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=64,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        shuffle=False,
        pin_memory=False, num_workers=16)

    net = net.to(device)
    net.eval()
    correct = 0
    total = 0

    y_test = []
    y_score = []

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            targets = targets.to('cpu').reshape(-1).numpy()
            one_hot_targets = np.eye(n_classes)[targets]
            y_test.append(one_hot_targets)
            y_score.append(outputs.to('cpu').numpy())

    acc = 100. * correct / total
    print("acc: ", acc)

    y_test = np.concatenate(y_test, axis=0)
    y_score = np.concatenate(y_score, axis=0)
    print(y_test.shape)
    print(y_score.shape)

    # Compute ROC curve and ROC area for each class

    lw = 2

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(y_score)
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'gold', 'darkgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(idx_to_class[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of each classes')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('roc.png')
