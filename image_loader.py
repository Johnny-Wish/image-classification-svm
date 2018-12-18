import os
import numpy as np
from PIL import Image


def is_valid_file(filename, suffixes):
    for suffix in suffixes:
        if filename.endswith(suffix):
            return True
    return False


class ImageLoader:
    def __init__(self, folder, width, height):
        self.folder = folder
        self.suffixes = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"]
        self.image_shape = (width, height)
        self._image_names = None
        self._images = None
        self._image_vectors = None
        self._image_matrix = None

    def _set_image_names(self):
        self._image_names = [f for f in os.listdir(self.folder) if is_valid_file(f, self.suffixes)]

    def _set_images(self):
        if self._image_names is None:
            self._set_image_names()
        self._images = [Image.open(os.path.join(self.folder, f)).resize(self.image_shape) for f in self._image_names]

    def _set_image_vectors(self):
        if self._images is None:
            self._set_images()
        self._image_vectors = [np.array(img.getdata()).reshape(-1) for img in self._images]

    def _set_image_matrix(self):
        if self._image_vectors is None:
            self._set_image_vectors()
        self._image_matrix = np.array(self._image_vectors)

    @property
    def image_names(self):
        if self._image_names is None:
            self._set_image_names()
        return self._image_names

    @property
    def images(self):
        if self._images is None:
            self._set_images()
        return self._images

    @property
    def image_vectors(self):
        if self._image_vectors is None:
            self._set_image_vectors()
        return self._image_vectors

    @property
    def image_matrix(self):
        if self._image_matrix is None:
            self._set_image_matrix()
        return self._image_matrix
