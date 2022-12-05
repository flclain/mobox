"""Get anchor trajectories with K-means."""
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from scipy.optimize import linear_sum_assignment

from config.defaults import get_cfg
from projects.wayformer.feature_generator import WaymoFeatureGenerator
from projects.wayformer.scenario_generator import WaymoScenarioGenerator

N = 500


def vis_tracks(tracks):
    plt.gca().set_aspect("equal")
    for t in tracks:
        plt.plot(t[:, 0], t[:, 1], 'r.')
    plt.show()


def get_anchors(M):
    cfg = get_cfg()
    fg = WaymoFeatureGenerator(cfg)
    sg = WaymoScenarioGenerator(cfg)
    idx = 0
    tracks = []
    for idx in range(N):
        scenario, track_id = sg.get(balance=True)
        print(idx, scenario)
        idx += 1
        if idx == N:
            break
        scenario.tracks = [x for x in scenario.tracks if x.filter(
            pl.col("is_valid")).shape[0] == len(scenario.timestamps)]
        if track_id not in scenario.focused_track_ids:
            continue
        agent_feats = fg.agent_feats(scenario, track_id)
        if agent_feats is None:
            continue
        tracks.append(agent_feats[10:, 0, :2])

    # K-means.
    Xs = np.stack(tracks, 0)  # [N,80,2]
    # torch.save(Xs, "tracks.pth")
    km = TimeSeriesKMeans(n_clusters=M, random_state=0).fit(Xs)
    centroids = km.cluster_centers_  # [M,T,2]
    return centroids


if __name__ == "__main__":
    anchors = get_anchors(M=64)
    torch.save(anchors, f"./cache/anchors.pth")
