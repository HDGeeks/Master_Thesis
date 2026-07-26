import sys


import modules.configuration as conf
import modules.data_processing as dp


def build_canonical_target_form(string):
    """Strip whitespace from a string and convert it to lower case.

    :param string: The string that should be processed.
    :return string: The processed string.
    """

    string = string.lower()
    string = string.strip()
    return string


def build_canonical_targets_map(targets):
    """Build a map that maps the canonical representation of a target to its
    original form.

    :param targets: A list of targets for which the map should be constructed.
    :return targets_map: The constructed map.
    """

    target_name = conf.config["prompt"]["target_name"]
    targets_map = {}

    for target in targets:
        canonical_target = dp.build_canonical_target_form(target)
        if canonical_target in targets_map.keys():
            print(f"The canonical {target_name} string \"{canonical_target}\" "
                  f"is the resulting string for the {target_name}s \""
                  f"{target}\" and \"{targets_map[canonical_target]}\". Such "
                  f"duplicates are not allowed. Exiting!")
            sys.exit(1)
        targets_map[canonical_target] = target

    return targets_map
