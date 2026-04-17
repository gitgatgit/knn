from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from knn import KNN, accuracy
from cross_val import NestedCrossVal
from confusion_matrix import metrics


def load_wine_data():
    wine = datasets.load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    intensity_col = next(c for c in df.columns if c.endswith('_intensity'))
    df = df.rename(columns={intensity_col: 'colour_intensity'})
    X = df[['alcohol', 'flavanoids', 'colour_intensity', 'ash']].values
    y = wine.target
    return X, y


if __name__ == '__main__':
    X, y = load_wine_data()
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNN(num_neighbours=3, distance='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f'Hold-out test accuracy: {accuracy(y_test, y_pred):.4f}')
    print(metrics(y_test, y_pred))

    print('\n--- Nested Cross-Validation ---')
    mean_acc, std, cm = NestedCrossVal(
        X, y, nFolds=5, listN=[1, 3, 5, 7],
        distances=['euclidean', 'manhattan'], mySeed=42
    )
    print(f'\nMean CV accuracy: {mean_acc} ± {std}')
    print(f'Confusion matrix (last fold):\n{cm}')
