import math
import numpy as np


def get_polyline_length(polyline):
    """Calculate the length of a polyline.

    Args:
      polyline: (np.array) polyline waypoints, sized [T,2] or [N,T,2].

    Returns:
      (float) length of the polyline.
    """
    offsets = np.diff(polyline, axis=-2)
    return np.linalg.norm(offsets, axis=-1).sum(-1)


def interp_arc(points, num_points):
    """Linearly interpolate equally-spaced points along a polyline.

    Args:
      polyline: (np.array) polyline waypoints, sized [N,2].
      num_points: (int) number of points that will be uniformly interpolated.

    Reference:
      https://github.com/argoai/av2-api/blob/main/src/av2/geometry/interpolate.py#L120
    """
    n, _ = points.shape
    eq_spaced_points = np.linspace(0, 1, num_points)

    chord_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    chord_len = chord_len / np.sum(chord_len)

    cum_arc = np.zeros(len(chord_len) + 1)
    cum_arc[1:] = np.cumsum(chord_len)

    bins = np.digitize(eq_spaced_points, bins=cum_arc).astype(int)
    bins[np.where((bins <= 0) | (eq_spaced_points <= 0))] = 1
    bins[np.where((bins >= n) | (eq_spaced_points >= 1))] = n - 1

    s = np.divide((eq_spaced_points - cum_arc[bins-1]), chord_len[bins-1])
    anchors = points[bins-1, :]
    offsets = (points[bins, :] - points[bins-1, :]) * s.reshape(-1, 1)
    points_interp = anchors + offsets
    return points_interp


def interp_polyline_by_fixed_interval(polyline, interval):
    """Resample waypoints of a polyline so that waypoints appear roughly at fixed intervals from the start.

    Args:
      polyline: (np.array) polyline waypoints, sized [N,2].
      interval: (float) space interval between waypoints, in meters.

    Returns:
      (np.array) interpolated polyline waypoints.

    Reference:
      https://github.com/argoai/av2-api/blob/main/src/av2/geometry/polyline_utils.py#L37
    """
    if len(polyline) < 2:  # at least has 2 points
        return polyline
    length = get_polyline_length(polyline)
    num_points = math.floor(length / interval) + 1
    return interp_arc(polyline, num_points)
