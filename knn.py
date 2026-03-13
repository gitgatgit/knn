from sklearn import datasets 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
# load data
wine=datasets.load_wine()
#print(wine.DESCR)

# this dataset has 13 features, we will only choose a subset of these
df_wine = pd.DataFrame(wine.data, columns = wine.feature_names )
intensity_feature = next(col for col in df_wine.columns if col.endswith('_intensity'))
df_wine = df_wine.rename(columns={intensity_feature: 'colour_intensity'})
selected_features = ['alcohol','flavanoids','colour_intensity','ash']

# extract the data as numpy arrays of features, X, and target, y
X = df_wine[selected_features].values
y = wine.target

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Preprocess data
X = StandardScaler().fit_transform(X)
# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


import numpy as np
from collections import Counter


class KNN():
    """
    K-nearest neighbour classifier 

    Attributes: ? 
    - slope (float): Slope of the regression line.
    - intercept (float): Intercept of the regression line.

    Methods:
    - fit(X, y) : Fit the model to input data.
    - euclidean() : 
    - manhattan() : 
    - neighbours(x) : 
    - predict(X) : Predict target values for new data.
    
    """


    def __init__(self, num_neighbours=3, distance='euclidean'):
        """Inits KNN
        Args:
            num_neighbours: k-number of neighbours
            distance: distance metric used, 'euclidean' 'manhattan'

        Returns
        ------
        - None
        """
        #initialise neighbours
        self.num_neighbours = num_neighbours
        #initialise distance
        self.distance = distance

    def fit(self, X, y):
        """
        Fits split dataset 

        Params
        ----------
        - X : array
        - y : array

        Returns
        --------
        - None
        """
        #Fit X
        self.X_train = X
        #Fit y
        self.y_train = y
        
    
    def euclidean(self, v1, v2):
        """
        Calculates euclidean distance between two vector points

        Params
        ------
        - v1 : 
        - v2 : 

        Returns
        ------
        - float : distance
        
        """

        #Calculate euclidean distance ()
        distance = np.sqrt(np.sum((v1 - v2)**2))
        
        return distance

    def manhattan(self, v1, v2):
        """
        Calculates manhattan distance between two vector points

        Params
        ------
        - v1 : 
        - v2 :

        Returns
        ------
        - float : distance
        
        """
        #Calculate manhattan distance ()
        diff = v1 - v2
        abs_diff = np.abs(diff)
        distance = np.sum(abs_diff)
        
        return distance

    def neighbours(self, x):
        """
        Sorts the neighbours according to the distance function

        Params
        ------
        - x : 

        Returns
        ------
        - list : sorted_neighbours
        
        """
        #Empty distances list
        distances = []

        #Iterate through X_train set, choose distance, calculate distances on each ... , append distances to list
        for x_train in self.X_train:
            if self.distance == 'euclidean':
                dist = self.euclidean(x, x_train)
                distances.append(dist)
            elif self.distance == 'manhattan':
                dist = self.manhattan(x, x_train)
                distances.append(dist)

        #Sort distances
        sorted_indices = np.argsort(distances)
        #Sort according to num_neighbours
        sorted_neighbours = [self.y_train[i] for i in sorted_indices[:self.num_neighbours]]

        return sorted_neighbours

    def predict(self, X):
        """
        Predicts the class neighbour belongs to

        Params
        ---------

        - X : 
        Returns
        ---------
        - array : predictions
        
        """
        #Empty predictions list
        predictions = []
        #Loop through neighbours, return most common value on each index
        for x in X:
            neighbours = self.neighbours(x)
            neighbour_counts = Counter(neighbours)
            top = neighbour_counts.most_common(1)[0][0]
            
            predictions.append(top)
            
        return np.array(predictions)


def accuracy(x, y):
    """
    Calculates accuracy

    Params
    ------
    - x : array
    - y : array

    Returns
    ------
    - float : accuracy
    
    """
    # Array length
    total = len(x)
    # Init counter
    counter = 0

    # count values in prediction that matches test set
    for i in range(total):
        if x[i] == y[i]:
            counter += 1
            
    # number of correct counts over length of array
    accuracy = counter / total

    return accuracy
