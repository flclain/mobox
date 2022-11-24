import numpy as np

from mobox.utils.coordinate import CoordinateTransform
from mobox.utils.polyline import interp_polyline_by_fixed_interval


class MapEncoder:
    """Encode map elements."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _split_polyline(self, points):
        """Split polyline into smaller segments with defined maximum length.

        Args:
          points: (np.array) polyline points, sized [N,2].

        Returns:
          (np.array) smaller segments, sized [M,L,2].

        Reference:
          https://github.com/nachiket92/PGP/blob/main/datasets/nuScenes/nuScenes_vector.py#L394
        """
        N = len(points)
        L = int(self.cfg.FEATURE.MAX_POLYLINE_LEN / self.cfg.FEATURE.POLYLINE_INTERP_INTERVAL)
        M = N // L if N % L == 0 else N // L + 1
        segments = np.zeros((M*L, 2))
        segments[:N] = points
        return segments.reshape(M, L, 2)

    def _encode_polyline(self, df_polyline, pos):
        """Encode polyline.

        It does the followings:
          1. Transform map element polyline points to target agent coordinate frame.
          2. Interpolate polylines.
          3. Split polylines into smaller segments with defined maximum length.
          4. Encode polyline as vector.

        Args:
          df_polyline: (pl.DataFrame) map element polyline.
          pos: (np.array) target agent pose of (origin_x, origin_y, heading), sized [1,3].

        Returns:
          (np.array) vector representation of polyline, sized [M,L,2].
        """
        INTERP_INTERVAL = self.cfg.FEATURE.POLYLINE_INTERP_INTERVAL
        points = df_polyline[["px", "py"]].to_numpy()  # [N,2]
        # Transform coordinates.
        points = CoordinateTransform.transform_points(points[None, :, :], pos)[0]
        # Interpolate.
        interp = interp_polyline_by_fixed_interval(points, interval=INTERP_INTERVAL)
        # Split into segments.
        segments = self._split_polyline(interp)  # [M,L,2]
        # Segments to vector.
        L = segments.shape[1]
        start_vec = segments[:, :L-1, :]
        end_vec = segments[:, 1:, :]
        return np.concatenate([start_vec, end_vec], axis=2)

    def _sort_polylines(self, polylines):
        """Sort map polylines by its middle point distance to target agent.

        Args:
          polylines: (np.array) map segment polylines, each sized [N,M,2].

        Returns:
          (np.array) sorted map segments.
        """
        lines = [x[x[:, 0] != 0] for x in polylines]
        middle_points = [0.5*(x[0]+x[-1]) for x in lines]
        dists = [np.hypot(x[0], x[1]) for x in middle_points]
        ids = np.argsort(dists)
        return polylines[ids]

    def encode(self, df_map, pos):
        """Encode scenario map to features.

        Args:
          df_map: (pl.DataFrame) map data frame.
          pos: (np.array) target agent pose of (origin_x, origin_y, heading), sized [1,3].

        Returns:
          (np.array) encoded map features, sized [num_segments, num_points, 4+1], the last dimension is valid mask.
        """
        M = self.cfg.FEATURE.MAX_NUM_MAP_FEATS
        feats = [self._encode_polyline(x, pos) for x in df_map.partition_by("id") if len(x) > 1]
        feats = np.concatenate(feats, 0)
        N, L, D = feats.shape

        # Sort map segments by distance to target.
        # feats = self._sort_polylines(feats)

        # Append valid mask.
        feats = np.concatenate([feats, np.ones((N, L, 1))], axis=-1)

        # Padding.
        feats = feats[:M, :, :] if N > M else np.concatenate(
            [feats, np.zeros((M-N, L, D+1))], axis=0)
        return feats
