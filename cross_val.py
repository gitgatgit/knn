import numpy as np
from knn import KNN, accuracy
from confusion_matrix import conf_mat


def NestedCrossVal(X, y, nFolds, listN, distances, mySeed):
    """
    Nested cross-validation with inner hyperparameter search over k and distance metric.

    Params
    ------
    X        : array, feature matrix
    y        : array, labels
    nFolds   : int, number of folds for both outer and inner CV
    listN    : list of int, candidate k values
    distances: list (unused — both euclidean and manhattan are always evaluated)
    mySeed   : int, random seed

    Returns
    -------
    mean_outer_accuracy : float
    deviation           : float
    confusion_matrix    : ndarray (last outer fold)
    """

    def kfold_indices(X, k):
        fold_size = len(X) // k
        np.random.seed(mySeed)
        indices = np.random.permutation(np.arange(len(X)))
        folds = []
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate(
                [indices[:i * fold_size], indices[(i + 1) * fold_size:]]
            )
            folds.append((train_indices, test_indices))
        return folds

    outer_fold_indices = kfold_indices(X, nFolds)
    mean_scores = []
    confusion_matrix = None

    for outer_train_indices, outer_test_indices in outer_fold_indices:
        X_outer_train = X[outer_train_indices]
        y_outer_train = y[outer_train_indices]
        X_outer_test = X[outer_test_indices]
        y_outer_test = y[outer_test_indices]

        inner_scores = []
        inner_fold_indices = kfold_indices(X_outer_train, nFolds)
        conf_list = []

        for inner_train_indices, inner_test_indices in inner_fold_indices:
            fold_scores_e = []
            fold_scores_m = []

            for k in listN:
                clf_e = KNN(num_neighbours=k, distance='euclidean')
                clf_e.fit(X_outer_train[inner_train_indices], y_outer_train[inner_train_indices])
                y_pred_e = clf_e.predict(X_outer_train[inner_test_indices])
                fold_scores_e.append((float(accuracy(y_outer_train[inner_test_indices], y_pred_e)), k))

                clf_m = KNN(num_neighbours=k, distance='manhattan')
                clf_m.fit(X_outer_train[inner_train_indices], y_outer_train[inner_train_indices])
                y_pred_m = clf_m.predict(X_outer_train[inner_test_indices])
                fold_scores_m.append((float(accuracy(y_outer_train[inner_test_indices], y_pred_m)), k))

            avg_e = np.mean([s for s, _ in fold_scores_e])
            avg_m = np.mean([s for s, _ in fold_scores_m])

            best_k_e = max(fold_scores_e, key=lambda x: x[0])
            top_neighbour_e = fold_scores_e.index(best_k_e) + 1

            best_k_m = max(fold_scores_m, key=lambda x: x[0])
            top_neighbour_m = fold_scores_m.index(best_k_m) + 1

            if avg_e >= avg_m:
                best_clf = KNN(num_neighbours=top_neighbour_e, distance='euclidean')
                optimal_k = top_neighbour_e
                best_dist = 'euclidean'
            else:
                best_clf = KNN(num_neighbours=top_neighbour_m, distance='manhattan')
                optimal_k = top_neighbour_m
                best_dist = 'manhattan'

            best_clf.fit(X_outer_train, y_outer_train)
            y_pred_best = best_clf.predict(X_outer_test)
            final_accuracy = accuracy(y_outer_test, y_pred_best)
            inner_scores.append(final_accuracy)

            confusion_matrix = conf_mat(y_outer_test, y_pred_best)
            conf_list.append(confusion_matrix)

        print(f'Final Accuracy: {np.round(final_accuracy, 6)}  k={optimal_k}  {best_dist}')

        mean_scores.append(np.mean(inner_scores))

    mean_outer_accuracy = np.mean(mean_scores)
    deviation = np.std(mean_scores)

    return np.round(mean_outer_accuracy, 6), np.round(deviation, 6), confusion_matrix
