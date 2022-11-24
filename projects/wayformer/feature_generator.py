import numpy as np
import polars as pl

from einops import rearrange

from mobox.encoder.map_encoder import MapEncoder
from mobox.encoder.agent_encoder import AgentEncoder


class WaymoFeatureGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.map_encoder = MapEncoder(cfg)
        self.agent_encoder = AgentEncoder(cfg)

    def agent_feats(self, scenario, track_id):
        """Contains a sequence of agent features.

        For each timestamp t, consider features that define the state of the agent.
        a.g. x, y, velocity, acceleration, bounding box, etc.

        Note the last feature dimension is valid mask.

        Args:
          scenario: (Scenario) input scenario.
          track_id: (int) target agent track id.

        Returns:
          (np.array) agent history, sized [T,1,D_h].
        """
        H = self.cfg.TRACK.HISTORY_SIZE
        df_agent = scenario.query_track_by_id(track_id)
        pos = df_agent[H, ["px", "py", "yaw"]].to_numpy()
        feats = self.agent_encoder.encode([df_agent], pos)  # [1,T,D]
        T, D = feats.shape[1:]
        return feats.reshape(T, 1, D).astype(np.float32)

    def nearby_agent_feats(self, scenario, track_id):
        """For each modeled agent, a fixed number of closest nearby agents around are considered.

        Note the last feature dimension is valid mask.

        Args:
          scenario: (Scenario) input scenario.
          track_id: (int) target agent track id.

        Returns:
          (np.array) nearby agent features, sized [T,S_i,D_i].
        """
        L = self.cfg.TRACK.SIZE
        H = self.cfg.TRACK.HISTORY_SIZE
        M = self.cfg.FEATURE.NUM_NEARBY_AGENTS
        R = self.cfg.FEATURE.AGENT_NEARBY_RADIUS
        track = scenario.query_track_by_id(track_id)
        pos = track[H, ["px", "py", "yaw"]].to_numpy()
        x, y = pos[0, :2]

        agents = [t for t in scenario.tracks if
                  t[0, "track_id"] != track_id and
                  t[0, "is_valid"] and
                  t[H, "is_valid"] and
                  (t[H, "px"] - x)**2 + (t[H, "py"] - y)**2 < R*R and
                  ((t["px"] - x)**2 + (t["py"] - y)**2 < 4*R*R).sum() == L]
        if len(agents) == 0:
            return None
        feats = self.agent_encoder.encode(agents, pos)
        feats = np.transpose(feats, (1, 0, 2))  # [N,T,D] -> [T,N,D]
        T, N, D = feats.shape
        feats = feats[:, :M, :] if N > M else np.concatenate([feats, np.zeros((T, M-N, D))], axis=1)
        return feats.astype(np.float32)

    def roadgraph(self, scenario, track_id):
        """The roadgraph contains road features around the agent.

        Note the last feature dimension is valid mask.

        Args:
          scenario: (Scenario) input scenario.
          track_id: (int) target agent track id.

        Returns:
          (np.array) encoded map features, sized [num_points, num_segments, D].
        """
        H = self.cfg.TRACK.HISTORY_SIZE
        R = self.cfg.FEATURE.AGENT_NEARBY_RADIUS
        track = scenario.query_track_by_id(track_id)
        pos = track[H, ["px", "py", "yaw"]].to_numpy()
        x, y = pos[0, :2]
        df_map = scenario.map.filter((pl.col("px")-x).pow(2) + (pl.col("py")-y).pow(2) < R * R)
        if len(df_map) == 0:
            return None
        feats = self.map_encoder.encode(df_map, pos)
        feats = rearrange(feats, "S T D -> T S D")
        return feats.astype(np.float32)

    def traffic_light_stats(self, scenario):
        """Contains the states of the traffic signals that are closest to the agent.

        Args:
          scenario: (Scenario) input scenario.

        Returns:
          (np.array) traffic light states, sized [T,S_tls,D_tls].
        """
        assert(False)


if __name__ == "__main__":
    from config.defaults import get_cfg
    from projects.wayformer.scenario_generator import WaymoScenarioGenerator
    cfg = get_cfg()
    sg = WaymoScenarioGenerator(cfg)
    fg = WaymoFeatureGenerator(cfg)
    i = 0
    for scenario in sg.scenarios:
        track_id = -1
        agent_feats = fg.agent_feats(scenario, track_id)
        nearby_feats = fg.nearby_agent_feats(scenario, track_id)
        map_feats = fg.roadgraph(scenario, track_id)
        print(agent_feats.shape)
        print(nearby_feats.shape)
        print(map_feats.shape)
        print(map_feats)
        break
