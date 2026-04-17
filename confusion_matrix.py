import numpy as np


def conf_mat(y_test, y_pred):
    """Build a confusion matrix from true and predicted labels."""
    classes = np.unique(np.concatenate([y_test, y_pred]))
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_test, y_pred):
        matrix[class_to_idx[true]][class_to_idx[pred]] += 1
    return matrix


def metrics(y_test, y_pred):
    """
    Calculate accuracy, per-class precision, and per-class recall.

    Returns
    -------
    accuracy : float
    precision : ndarray, per class
    recall : ndarray, per class
    """
    total = len(y_test)
    counter = sum(1 for t, p in zip(y_test, y_pred) if t == p)
    accuracy = counter / total

    confusion_matrix = conf_mat(y_test, y_pred)

    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

    precision = np.nan_to_num(
        np.divide(true_positives, true_positives + false_positives)
    )
    recall = np.nan_to_num(
        np.divide(true_positives, true_positives + false_negatives)
    )

    return f'accuracy: {accuracy}', f'precision: {precision}', f'recall: {recall}'
