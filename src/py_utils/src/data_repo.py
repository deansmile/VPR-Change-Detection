import json
import os

import dill
import matplotlib.pyplot as plt
import numpy as np

##########
# reader #
##########


def _np_read(path):
    return np.load(path, allow_pickle=True)


def _dill_read(path):
    with open(path, "rb") as fd:
        return dill.load(fd)


def _img_read(path):
    return plt.imread(path)


def reader_factory(path):

    reader = {
        "npy": _np_read,
        "dill": _dill_read,
        "png": _img_read,
    }

    extension = path.split(".")[-1]
    if extension not in reader:
        raise RuntimeError(f"Do not know how to handle {path}")
    return reader[extension]


##########
# writer #
##########


def _np_write(path, obj):
    # numpy array can be written by dill
    # but np.save is three times efficient than dill
    # so numpy array is handed separately
    np.save(path, obj)


def _dill_write(path, obj):

    with open(path, "wb") as fd:
        dill.dump(obj, fd)


def _img_write(path, obj):
    plt.imsave(path, obj)


def normalize_path_with_obj(path, obj, suggestion=None):

    # take suggestion as highest priority (manual setting)
    if suggestion is not None:
        return path + suggestion

    # otherwise, determine by obj itself
    extension = ".dill"

    if isinstance(obj, np.ndarray):
        extension = ".npy"

    if not path.endswith(extension):
        path += extension
    return path


def writer_factory(path):

    writer = {
        "npy": _np_write,
        "dill": _dill_write,
        "png": _img_write,
    }

    extension = path.split(".")[-1]
    if extension not in writer:
        raise RuntimeError(f"Do not know how to handle {path}")
    return writer[extension]


#############
# Data Repo #
#############


def serialize_object(obj, path, inplace=False, suggestion=None):
    """
    Serialize objects to a given path.
    """
    path = normalize_path_with_obj(path, obj, suggestion=suggestion)
    if os.path.exists(path) and not inplace:
        raise FileExistsError

    writer = writer_factory(path)
    writer(path, obj)
    return path


def deserialize_object(path):
    """
    Deserialize objects from a given path.
    """
    reader = reader_factory(path)
    return reader(path)


class DataRepositoryObserver:
    # TODO: consider using hash to manage attributes, it will give more
    # flexibility to serialize attributes

    def __init__(self, repo_dir):
        """
        Initialize the observer with a repository directory.
        """
        repo_dir = os.path.realpath(repo_dir)
        self.repo_dir = repo_dir
        self.metadata_file = os.path.join(repo_dir, "metadata.json")
        if not os.path.exists(self.metadata_file):
            self._save_metadata({})

    def __contains__(self, name):
        metadata = self._load_metadata()
        return name in metadata

    def add_item(self, name, obj, inplace=False, extension=None, **attributes):
        """
        Add an item with attributes to the repository.
        """

        path = os.path.join(self.repo_dir, name)
        path = serialize_object(
            obj, path, inplace=inplace, suggestion=extension
        )

        name = os.path.relpath(path, start=self.repo_dir)

        metadata = self._load_metadata()
        metadata[name] = attributes
        self._save_metadata(metadata)

    def del_item(self, name):
        metadata = self._load_metadata()
        if name not in self:
            raise FileNotFoundError(f"{name} is not in metadata.")

        path = os.path.join(self.repo_dir, name)
        metadata.pop(name)
        os.remove(path)
        self._save_metadata(metadata)

    def get_item(self, name):
        metadata = self._load_metadata()
        if name not in metadata:
            raise FileNotFoundError(f"{name} is not in metadata.")

        path = os.path.join(self.repo_dir, name)
        obj = deserialize_object(path)
        return obj

    def get_attribute(self, name):
        metadata = self._load_metadata()
        if name not in metadata:
            raise FileNotFoundError(f"{name} is not in metadata.")

        return metadata[name]

    def list_items(self, filter_by=None):
        """
        List items in the repository,
        optionally filtering by a custom function.
        The filter_by should take two arguments (name, attributes)
        and return a boolean.
        """
        metadata = self._load_metadata()

        if filter_by is None:
            return metadata

        filtered_metadata = {}
        for name, attrs in metadata.items():
            if not filter_by(name, attrs):
                continue
            filtered_metadata[name] = attrs

        return filtered_metadata

    def _load_metadata(self):
        """
        Load metadata from the metadata file.
        """
        with open(self.metadata_file, "r") as file:
            return json.load(file)

    def _save_metadata(self, metadata):
        """
        Save metadata to the metadata file.
        """
        with open(self.metadata_file, "w") as file:
            json.dump(metadata, file, indent=4)
