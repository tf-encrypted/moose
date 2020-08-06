from collections import ChainMap

import yaml


def load_cluster_spec(filename):
    with open(filename) as file:
        clusters_spec = yaml.load(file, Loader=yaml.FullLoader)
    return dict(ChainMap(*clusters_spec["workers"]))
