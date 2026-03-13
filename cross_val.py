import knn

def NestedCrossVal(X, y, nFolds, listN, distances, mySeed):
    """
    Nested cross validation

    Params 
    ------

    - X : array
    - y : array
    - nFolds : int
    - listN : list
    - distances : list 
    - mySeed : int

    Returns
    ------
    - float : accuracy
    - array : confusion_matrix   
    
    """
    

    def kfold_indices(X, k):

        """

        Split dataset to indices

        Params
        ------
        - X : array

        Returns
        ------
        - list : folds

        """
        
        fold_size = len(X) // k
        np.random.seed(mySeed)
        indices = np.random.permutation(np.arange(0, len(X), 1))
        folds = []
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            folds.append((train_indices, test_indices))
        return folds

    outer_k = nFolds
    inner_k = nFolds

    outer_fold_indices = kfold_indices(X, outer_k)
    mean_scores = []

    

    for outer_train_indices, outer_test_indices in outer_fold_indices:
        
        X_outer_train, y_outer_train = X[outer_train_indices], y[outer_train_indices]
        X_outer_test, y_outer_test = X[outer_test_indices], y[outer_test_indices]

        inner_scores = []
        inner_fold_indices = kfold_indices(X_outer_train, inner_k)
        
        conf_list = []
        
        for inner_train_indices, inner_test_indices in inner_fold_indices:
            
            fold_scores_k = []
            fold_scores_k_m = []

            # Calculating scores for different values of k and distances
            for k in listN:
                classifier = KNN(num_neighbours=k, distance='euclidean')
                classifier.fit(X_outer_train[inner_train_indices], y_outer_train[inner_train_indices])
                y_pred = classifier.predict(X_outer_train[inner_test_indices])
                scoreA = accuracy(y_outer_train[inner_test_indices], y_pred)
                fold_scores_k.append((float(scoreA), k))

                classifierM = KNN(num_neighbours=k, distance='manhattan')
                classifierM.fit(X_outer_train[inner_train_indices], y_outer_train[inner_train_indices])
                y_predM = classifierM.predict(X_outer_train[inner_test_indices])
                scoreAM = accuracy(y_outer_train[inner_test_indices], y_predM)
                fold_scores_k_m.append((float(scoreAM), k))


            #average scores
            avg_E = np.mean([i[0] for i in fold_scores_k])
            avg_M = np.mean([i[0] for i in fold_scores_k_m])

            #top k values and neighbour
            best_k = max(fold_scores_k, key=lambda x: x[0])
            top_neighbour_E = fold_scores_k.index(best_k) + 1
            
            best_k_m = max(fold_scores_k_m, key=lambda x: x[0])
            top_neighbour_M = fold_scores_k_m.index(best_k_m) + 1

            #############################################

            #print(avg_E, avg_M)

            best = max(avg_E, avg_M)
            #print(best)
            classifier_best = KNN()
            # if top average score is equal to max average euclidean
            if best == avg_E:
                classifier_best = KNN(num_neighbours=top_neighbour_E, distance='euclidean')
            elif best == avg_M:
                classifier_best = KNN(num_neighbours=top_neighbour_M, distance='manhattan')
    
            #train classifier outer fold best values
            classifier_best.fit(X_outer_train, y_outer_train)
            y_pred_best = classifier_best.predict(X_outer_test)
            final_accuracy = accuracy(y_outer_test, y_pred_best)  
            inner_scores.append(final_accuracy)

            optimal_k = max(top_neighbour_E, top_neighbour_M)
        

            confusion_matrix = conf_mat(y_outer_test, y_pred_best)
            conf_list.append(confusion_matrix)
            
        #
        if optimal_k == top_neighbour_E:
            print('Final Accuracy:', np.round(final_accuracy, 6), optimal_k, 'euclidean')
        else:
            print('Final Accuracy:', np.round(final_accuracy, 6), optimal_k, 'manhattan')

        # Calculate the mean accuracy across all inner folds
        mean_inner_accuracy = np.mean(inner_scores)
        mean_scores.append(mean_inner_accuracy)
        

    mean_outer_accuracy = np.mean(mean_scores)
    deviation = np.std(mean_scores)

    
    
    return np.round(mean_outer_accuracy, 6), np.round(deviation, 6), confusion_matrix#, confusion_matrix
    
