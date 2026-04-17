import numpy as np
from sklearn.preprocessing import StandardScaler
from knn import KNN


class PlayerMatchmaker:
    """
    KNN-based player matchmaking engine.

    Finds the K most similar players from a pool based on gameplay stats.
    Returns ranked matches with a similarity score in [0, 1].
    """

    def __init__(self, k=5, distance='euclidean'):
        self.k = k
        self.distance = distance
        self.scaler = StandardScaler()
        self.knn = None
        self.player_ids = None

    def fit(self, player_ids, feature_matrix):
        """
        Register the player pool.

        Parameters
        ----------
        player_ids     : list of str/int, unique player identifiers
        feature_matrix : 2D array (n_players, n_features)
        """
        self.player_ids = np.array(player_ids)
        X_scaled = self.scaler.fit_transform(feature_matrix)
        self.X_scaled = X_scaled
        # request k+1 so we can drop self-matches (exact distance 0)
        self.knn = KNN(num_neighbours=self.k + 1, distance=self.distance)
        self.knn.fit(X_scaled, np.arange(len(player_ids)))

    def find_matches(self, player_features):
        """
        Find the top-K most similar players for a given feature vector.

        Parameters
        ----------
        player_features : 1D array of feature values (same order as fit)

        Returns
        -------
        list of (player_id, similarity_score) tuples, best match first.
        Similarity score is in (0, 1]; 1 = identical profile.
        """
        x = self.scaler.transform([player_features])[0]
        neighbors = self.knn.find_neighbors(x)

        results = []
        for idx, dist in neighbors:
            if dist < 1e-9:  # skip exact self-match
                continue
            similarity = round(1.0 / (1.0 + dist), 4)
            results.append((self.player_ids[idx], similarity))

        return results[:self.k]


if __name__ == '__main__':
    # Synthetic player pool
    # Features: [elo, win_rate, kd_ratio, avg_session_min, preferred_mode]
    #   preferred_mode: 0=casual  1=ranked  2=tournament
    np.random.seed(42)
    n = 40

    player_ids = [f'Player_{i:03d}' for i in range(n)]
    elo          = np.random.randint(800, 2800, n).astype(float)
    win_rate     = np.clip(np.random.normal(0.50, 0.15, n), 0.10, 0.90)
    kd_ratio     = np.clip(np.random.normal(1.0,  0.50, n), 0.10, 5.00)
    session_min  = np.random.randint(20, 180, n).astype(float)
    mode         = np.random.randint(0, 3, n).astype(float)

    feature_matrix = np.column_stack([elo, win_rate, kd_ratio, session_min, mode])
    feature_names  = ['elo', 'win_rate', 'kd_ratio', 'session_min', 'preferred_mode']

    matchmaker = PlayerMatchmaker(k=5, distance='euclidean')
    matchmaker.fit(player_ids, feature_matrix)

    # Find matches for a new mid-tier ranked player
    new_player = np.array([1500.0, 0.52, 1.1, 60.0, 1.0])
    mode_labels = {0: 'Casual', 1: 'Ranked', 2: 'Tournament'}

    print('=== Player Matchmaking Demo ===')
    print(
        f'Searching for:  Elo={new_player[0]:.0f}  '
        f'Win%={new_player[1]*100:.0f}%  '
        f'K/D={new_player[2]:.1f}  '
        f'Session={new_player[3]:.0f}min  '
        f'Mode={mode_labels[int(new_player[4])]}'
    )
    print()

    matches = matchmaker.find_matches(new_player)
    print(f'{"Rank":<5} {"Player":<14} {"Similarity":>10}  Profile')
    print('-' * 65)
    for rank, (pid, score) in enumerate(matches, 1):
        idx = list(player_ids).index(pid)
        print(
            f'{rank:<5} {pid:<14} {score:>10.4f}  '
            f'Elo={elo[idx]:.0f}  '
            f'Win%={win_rate[idx]*100:.0f}%  '
            f'K/D={kd_ratio[idx]:.1f}  '
            f'Mode={mode_labels[int(mode[idx])]}'
        )
