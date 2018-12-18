import numpy as np
from sklearn.preprocessing import LabelEncoder
from image_loader import ImageLoader


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

    def shuffle(self):
        p = np.random.permutation(len(self._X))
        self._X = self._X[p]
        self._y = self._y[p]

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

    @classmethod
    def from_subsets(cls, train, test):
        instance = cls.__new__()
        instance._train = train
        instance._test = test
        return instance

    def shuffle_train(self):
        self._train.shuffle()

    def shuffle_test(self):
        self._test.shuffle()

    def shuffle(self):
        self.shuffle_train()
        self.shuffle_test()


class ImageDataset(BaseDataset):
    @staticmethod
    def get_2d_array(img):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 1:
                return np.stack([img])  # treat the input as a single data point
            elif len(img.shape) == 2:
                return img  # treat the input as a n_sample x n_dimensional matrix
            elif len(img.shape) >= 3:
                return img.reshape([img.shape[0], -1])  # treat as an n_sample x width x height (x channel) tensor

        elif isinstance(img, ImageLoader):
            return img.image_matrix

    def __init__(self, train0, test0, train1, test1):
        super(ImageDataset, self).__init__()

        train0 = ImageDataset.get_2d_array(train0)
        train1 = ImageDataset.get_2d_array(train1)
        X_train = np.concatenate([train0, train1])
        y_train = np.concatenate([np.zeros(len(train0)), np.ones(len(train1))])
        self._train = Subset(X_train, y_train)

        test0 = ImageDataset.get_2d_array(test0)
        test1 = ImageDataset.get_2d_array(test1)
        X_test = np.concatenate([test0, test1])
        y_test = np.concatenate([np.zeros(len(test0)), np.zeros(len(test1))])
        self._test = Subset(X_test, y_test)
