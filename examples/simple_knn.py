import logging
import numpy as np

from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator


LOG_FORMAT = f"%(asctime)s %(name)s [%(levelname)s] %(message)s"


class SimpleKNN(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.users = None
        self.items = None
        self.interactions = None
        self.scores = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.set_logger()

    def fit(self, data: DataFrame) -> 'SimpleKNN':
        assert np.isin(["user_id", "item_id", "score"], data.columns).all()
        self.users, user_indexes = np.unique(data["user_id"], return_inverse=True)
        self.logger.info(f"Found {len(self.users)} unique users")

        self.items, item_indexes = np.unique(data["item_id"], return_inverse=True)
        self.logger.info(f"Found {len(self.items)} unique items")

        self.interactions = csr_matrix((data["score"], (user_indexes, item_indexes)))
        self.logger.info(f"Create user-item interactions matrix of shape {self.interactions.shape}")

        self.scores = self.interactions.dot(self.interactions.T.dot(self.interactions)).toarray()
        self.logger.info("Evaluate scores")

        self.scores = self.scores * (1 - self.interactions.toarray())
        self.logger.info("Remove seen")
        return self

    def predict(self, k: int) -> DataFrame:
        assert k <= len(self.items)
        recommendations = self.items[np.argsort(self.scores, axis=1)[:, -k:]][:, ::-1].tolist()
        self.logger.info(f"Got {k} recommendations for each user")

        recommendations = DataFrame({"user_id": self.users, "item_id": recommendations})
        recommendations = recommendations.explode("item_id").assign(
            rank=lambda x: x.groupby("user_id").cumcount() + 1
        )
        self.logger.info("Pack recommendations into a pandas.DataFrame")
        return recommendations

    def set_logger(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)
