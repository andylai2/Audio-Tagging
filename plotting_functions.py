import itertools
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics


def plot_confusion_matrix(true_labels,
                          predicted_labels,
                          labels_list,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """ Operates on two lists, true_labels and predicted_labels.
    Not 100% sure yet how this would be used, so keeping the input format
    relatively general for now.
    
    labels_list: a list of all the different possible labels.
    normalize: if False, matrix contains prediction results as counts.
        If true, matrix contrains results as ratios.
    title: Title of plot
    cmap: Color of plot

    """
    cm = sklearn.metrics.confusion_matrix(
            true_labels, predicted_labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels_list))
    plt.xticks(tick_marks, labels_list, rotation=45)
    plt.yticks(tick_marks, labels_list)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_error_per_epoch(val_accuracy_over_epochs,
                         title='Test data classification accuracy vs. epoch'):
    """Takes a list of validation accuracies, and makes a plot
    """
    plt.figure()
    plt.axis('square')
    plt.plot(val_accuracy_over_epochs)
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy %')
    plt.title(title)


# The following should be trivial TBH, so I don't think we
# really need to use Python for those.

def plot_input_image(image,title='Image representation of sound'):
    pass
    #plt.figure()
    #plt.imshow(image)
    #plt.title(title)

def mfcc_vs_spectrogram():
    pass

def time_dilation_vs_orig():
    pass
