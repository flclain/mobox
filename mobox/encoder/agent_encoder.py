import numpy as np

from mobox.utils.coordinate import CoordinateTransform


AGENT_FEAT_COLUMNS = ["px", "py", "yaw", "vx", "vy", "is_valid"]


class AgentEncoder:
    """Encode target agent and context agents.

    It operates as follows:
      1. Extract corresponding features from track data frame.
      2. Transform features to target coordinate frame.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, df_agents, pos):
        """Encode scenario agents to features.

        Args:
          df_agents: ([pl.DataFrame]) list of agent data frame, sized N.
          pos: (np.array) target coordinate frame of (origin_x, origin_y, heading), sized [1,3].

        Returns:
          (np.array) agent features, sized [N,T,D].
        """
        # Extract features.
        feats = [x[AGENT_FEAT_COLUMNS].to_numpy() for x in df_agents]
        feats = np.stack(feats, axis=0)  # [N,T,D]

        # Transform coordinate frame.
        points = feats[:, :, :2]
        yaw = feats[:, :, 2]
        vel = feats[:, :, 3:5]
        valid = feats[:, :, -1]

        points = CoordinateTransform.transform_points(points, pos)
        yaw = CoordinateTransform.transform_angles(yaw, pos)
        vel = CoordinateTransform.transform_vector(vel, pos)

        feats = np.concatenate([points, yaw[:, :, None], vel, valid[:, :, None]], axis=2)  # [N,T,D]
        feats[feats[:, :, -1] == 0] = 0  # set invalid feature to 0

        start_vec = feats[:, :-1, :2]
        end_vec = feats[:, 1:, :]
        feats = np.concatenate([start_vec, end_vec], axis=2)
        return feats
