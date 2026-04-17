import numpy as np
from sklearn.preprocessing import StandardScaler
from knn import KNN


class ContentRecommender:
    """
    KNN-based content recommender using cosine similarity.

    Given an item's feature vector, returns the K most similar items
    from the catalog. Works for movies, music, products, etc.
    """

    def __init__(self, k=5, distance='cosine'):
        self.k = k
        self.distance = distance
        self.scaler = StandardScaler()
        self.knn = None
        self.item_names = None

    def fit(self, item_names, feature_matrix):
        """
        Index the item catalog.

        Parameters
        ----------
        item_names     : list of str, item identifiers
        feature_matrix : 2D array (n_items, n_features)
        """
        self.item_names = np.array(item_names)
        X_scaled = self.scaler.fit_transform(feature_matrix)
        self.X_scaled = X_scaled
        # request k+1 neighbors so we can drop the seed item itself
        self.knn = KNN(num_neighbours=self.k + 1, distance=self.distance)
        self.knn.fit(X_scaled, np.arange(len(item_names)))

    def get_recommendations(self, item_features):
        """
        Return top-K items most similar to the given feature vector.

        Parameters
        ----------
        item_features : 1D array of feature values (same order as fit)

        Returns
        -------
        list of (item_name, similarity_score) tuples, best match first.
        Similarity score is in (0, 1]; 1 = identical profile.
        """
        x = self.scaler.transform([item_features])[0]
        neighbors = self.knn.find_neighbors(x)

        results = []
        for idx, dist in neighbors:
            if dist < 1e-9:  # skip exact self-match (seed item in catalog)
                continue
            similarity = round(1.0 / (1.0 + dist), 4)
            results.append((self.item_names[idx], similarity))

        return results[:self.k]


# fmt: off
MOVIES = [
    # (title, action, comedy, drama, sci_fi, romance, rating, year)
    ('Interstellar',       0, 0, 1, 1, 0, 8.6, 2014),
    ('The Martian',        0, 1, 0, 1, 0, 8.0, 2015),
    ('Gravity',            0, 0, 1, 1, 0, 7.7, 2013),
    ('Arrival',            0, 0, 1, 1, 0, 7.9, 2016),
    ('Ex Machina',         0, 0, 1, 1, 0, 7.7, 2014),
    ('Blade Runner 2049',  0, 0, 1, 1, 0, 8.0, 2017),
    ('Mad Max: Fury Road', 1, 0, 0, 0, 0, 8.1, 2015),
    ('John Wick',          1, 0, 0, 0, 0, 7.4, 2014),
    ('Avengers: Endgame',  1, 0, 0, 1, 0, 8.4, 2019),
    ('The Dark Knight',    1, 0, 1, 0, 0, 9.0, 2008),
    ('Inception',          1, 0, 1, 1, 0, 8.8, 2010),
    ('Grand Budapest Hotel', 0, 1, 0, 0, 0, 8.1, 2014),
    ('Superbad',           0, 1, 0, 0, 0, 7.6, 2007),
    ('The Hangover',       0, 1, 0, 0, 0, 7.7, 2009),
    ('Crazy Rich Asians',  0, 1, 0, 0, 1, 6.9, 2018),
    ('La La Land',         0, 0, 1, 0, 1, 8.0, 2016),
    ('Titanic',            0, 0, 1, 0, 1, 7.8, 1997),
    ('Pride & Prejudice',  0, 0, 1, 0, 1, 7.8, 2005),
    ('The Notebook',       0, 0, 1, 0, 1, 7.8, 2004),
    ('Forrest Gump',       0, 0, 1, 0, 0, 8.8, 1994),
]
# fmt: on

FEATURE_NAMES = ['action', 'comedy', 'drama', 'sci_fi', 'romance', 'rating', 'year']


def _genre_tags(feat):
    genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance']
    return ', '.join(g for g, f in zip(genres, feat) if f)


if __name__ == '__main__':
    titles = [m[0] for m in MOVIES]
    feature_matrix = np.array([[*m[1:]] for m in MOVIES], dtype=float)

    recommender = ContentRecommender(k=5, distance='cosine')
    recommender.fit(titles, feature_matrix)

    seed = 'Interstellar'
    seed_features = feature_matrix[titles.index(seed)]

    print('=== Movie Recommendation Demo ===')
    print(f'Because you watched: {seed}')
    print()

    recs = recommender.get_recommendations(seed_features)
    print(f'{"Rank":<5} {"Title":<25} {"Similarity":>10}  Genres (Year)')
    print('-' * 65)
    for rank, (title, score) in enumerate(recs, 1):
        idx = titles.index(title)
        feat = feature_matrix[idx]
        print(
            f'{rank:<5} {title:<25} {score:>10.4f}  '
            f'{_genre_tags(feat)} ({int(feat[6])})'
        )
