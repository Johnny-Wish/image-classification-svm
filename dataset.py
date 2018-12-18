import numpy as np
from sklearn.preprocessing import LabelEncoder


class Subset:
    def __init__(self, X=None, y=None, encoder=None):
        """
        A object simulating the training set / testing test / validation set of a super dataset
        :param X: A 2-dim sequence, n_samples x n_feature_dims
        :param y: A 1-dim/2-dim sequence, n_samples or n_samples x 1
        :param encoder: a fitted LabelEncoder instance or a callable that returns an encoded label vector
        """
        if X is None:
            self._X = []
        else:
            self._X = X

        if y is None:
            self._y = []
        else:
            self._y = y

        if isinstance(self._y, np.ndarray) and len(self._y.shape) == 2:
            self._y = self._y.flatten()  # the method np.ndarray.flatten() is stupid and doesn't update `self`

        # if an encoder is given, encode labels accordingly
        if isinstance(encoder, LabelEncoder):
            self._y = encoder.transform(self._y)
        elif callable(encoder):
            self._y = encoder(self._y)

        assert len(self._X) == len(self._y), "X and y differ in length {} != {}".format(len(self._X), len(self._y))

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    def __dict__(self):
        return {"X": self._X, "y": self._y}

    @classmethod
    def from_dict(cls, d):
        return cls(
            X=d.get("X", None),
            y=d.get("y", None),
        )

    def __repr__(self):
        return "<Subset: X={}, y={}>".format(self._X, self._y)


class BaseDataset:
    def __init__(self, *args, **kwargs):
        self._train = None  # type: Subset
        self._test = None  # type: Subset

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test
