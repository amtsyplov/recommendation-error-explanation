import numpy as np

from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator


class SimpleKNN(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.users = None
        self.items = None
        self.interactions = None
        self.scores = None

    def fit(self, data: DataFrame) -> 'SimpleKNN':
        assert np.isin(["user_id", "item_id", "score"], data.columns).all()
        self.users, user_indexes = np.unique(data["user_id"], return_inverse=True)
        self.items, item_indexes = np.unique(data["item_id"], return_inverse=True)
        self.interactions = csr_matrix((data["score"], (user_indexes, item_indexes)))
        self.scores = self.interactions.dot(self.interactions.T.dot(self.interactions)).toarray()
        self.scores = self.scores * (1 - self.interactions.toarray())
        return self

    def predict(self, k: int) -> DataFrame:
        assert k <= len(self.items)
        recommendations = self.items[np.argsort(self.scores, axis=1)[:, -k:]][:, ::-1].tolist()
        recommendations = DataFrame({"user_id": self.users, "item_id": recommendations})
        recommendations = recommendations.explode("item_id").assign(
            rank=lambda x: x.groupby("user_id").cumcount() + 1
        )
        return recommendations
