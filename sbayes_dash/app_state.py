from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import colors as mpl_colors


@dataclass
class AppState:

    clusters_path = None
    _clusters = None
    fig = None
    lines = None
    scatter = None
    cluster_colors = None
    locations = None
    object_data = None
    objects = None
    i_sample = 0
    burnin = 0

    slider = None



    @property
    def clusters(self):
        return self._clusters

    @clusters.setter
    def clusters(self, clusters):
        self._clusters = clusters
        self.cluster_colors = self.get_cluster_colors(self.n_clusters)

    @staticmethod
    def get_cluster_colors(K):
        # cm = plt.get_cmap('gist_rainbow')
        # cluster_colors = [colors.to_hex(c) for c in cm(np.linspace(0, 1, K, endpoint=False))]
        colors = []
        for i, x in enumerate(np.linspace(0, 1, K, endpoint=False)):
            b = i % 2
            h = x % 1
            s = 0.6 + 0.4 * b
            v = 0.5 + 0.3 * (1 - b)
            colors.append(
                mpl_colors.to_hex(mpl_colors.hsv_to_rgb((h, s, v)))
            )
        return colors

    @property
    def n_clusters(self) -> int:
        return self.clusters.shape[0]

    @property
    def n_samples(self) -> int:
        return self.clusters.shape[1]
