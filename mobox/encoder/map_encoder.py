import numpy as np

from mobox.utils.coordinate import CoordinateTransform
from mobox.utils.polyline import interp_polyline_by_fixed_interval


class MapEncoder:
    """Encode map elements."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _split_polyline(self, df_polyline):
        """Split polyline into smaller segments with defined maximum length.

        Args:
          df_polyline: (pl.DataFrame) map element polyline.

        Returns:
          (np.array) smaller segments, sized [M,L,3], the last dimension is valid mask.

        Reference:
          https://github.com/nachiket92/PGP/blob/main/datasets/nuScenes/nuScenes_vector.py#L394
        """
        points = df_polyline[["px", "py"]].to_numpy()  # [N,2]

        # Interpolate.
        INTERP_INTERVAL = self.cfg.FEATURE.POLYLINE_INTERP_INTERVAL
        interp = interp_polyline_by_fixed_interval(points, interval=INTERP_INTERVAL)

        # Split.
        N = len(interp)
        L = int(self.cfg.FEATURE.MAX_POLYLINE_LEN / self.cfg.FEATURE.POLYLINE_INTERP_INTERVAL)
        M = N // L if N % L == 0 else N // L + 1
        segments = np.zeros((M*L, 3))
        segments[:N, :2] = interp
        segments[:N, 2] = 1
        return segments.reshape(M, L, 3)

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
          (np.array) encoded map features, sized [S,L,D], S=num_segments, L=num_polyline_points.
        """
        segments = [self._split_polyline(x) for x in df_map.partition_by("id") if len(x) > 1]
        segments = np.concatenate(segments, 0)  # [S,L,D]

        # Sort map segments by distance to target.
        # segments = self._sort_polylines(segments)

        # Transform coordinates.
        points = CoordinateTransform.transform_points(segments[:, :, :2], pos)
        points[segments[:, :, 2] == 0] = 0

        start_vec = points[:, :-1, :]
        end_vec = points[:, 1:, :]
        feats = np.concatenate([start_vec, end_vec], axis=2)
        return feats
