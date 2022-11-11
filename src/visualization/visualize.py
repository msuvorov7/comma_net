import itertools

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_loss: list, val_loss: list):
    plt.figure(figsize=(16, 8))
    plt.plot(train_loss, marker='s', label='Train Loss')
    plt.plot(val_loss, marker='s', label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('mse_loss.jpg')


def plot_acc(train_acc: list, val_acc: list):
    plt.figure(figsize=(16, 8))
    plt.plot(train_acc, marker='s', label='Train ACC')
    plt.plot(val_acc, marker='s', label='Validation ACC')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.savefig('acc.jpg')


def plot_conf_matrix(cm: np.ndarray, target_names: list):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Test Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True value')
    plt.tight_layout()
    plt.savefig('conf_matrix.jpg')
