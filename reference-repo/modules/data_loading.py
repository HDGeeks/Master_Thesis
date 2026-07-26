import sys
import os
import json


import modules.configuration as conf


def get_abspath(path_spec):
    """Create the absolute path for a given path.

    Uses the tool base directory as starting point for relative paths. If an
    absolute path is provided, the function returns this path. If a relative
    path is provided, the function extracts the tool directory and concatenates
    the given relative path to it.

    :param path_spec: A relative or absolute path specification.
    """

    relative_starting_point = conf.tooldir
    absolute_path = os.path.join(relative_starting_point, path_spec)
    return absolute_path


def load_publications():
    """Load the configured publication metadata.

    This function takes metadata file path from the configuration, loads the
    data and return it.
    """

    metadata_json = conf.config["dataset"]["metadata_json"]
    metadata_json = get_abspath(metadata_json)

    if not os.path.isfile(metadata_json):
        print("Metadata file does not exists. Exiting!")
        sys.exit(1)

    with open(metadata_json, "r") as jsonfile:
        publications = json.load(jsonfile)
    return publications


def load_targets():
    """Load the configured targets file.

    This function takes targets file path from the configuration, loads the
    data and return it.
    """

    targets_json = conf.config["dataset"]["targets_json"]
    targets_json = get_abspath(targets_json)

    if not os.path.isfile(targets_json):
        print("Targets file does not exists. Exiting!")
        sys.exit(1)

    with open(targets_json, "r") as jsonfile:
        targets = json.load(jsonfile)
    return targets


def load_data():
    """Load the configured metadata and targets data."""

    publications = load_publications()
    targets = load_targets()
    return publications, targets


def load_annotated_metadata():
    """Load the configured annotated metadata file.

    This function takes annotated metadata file path from the configuration,
    loads the data and return it.
    """

    annotated_metadata_json = conf.config["dataset"]["annotated_json"]
    annotated_metadata_json = get_abspath(annotated_metadata_json)

    if not os.path.isfile(annotated_metadata_json):
        print("Annotated metadata file does not exists. Exiting!")
        sys.exit(1)

    with open(annotated_metadata_json, "r") as jsonfile:
        annotated_metadata = json.load(jsonfile)
    return annotated_metadata
