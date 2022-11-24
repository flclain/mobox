import numpy as np
import polars as pl

from config.defaults import get_cfg
from mobox.utils.polyline import get_polyline_length
from mobox.utils.geometry import rotate, vector_angle_diff

cfg = get_cfg()

# Orient class params.
STATIC_MAX_DIST = cfg.META.STATIC_MAX_DIST
STATIC_MAX_SPEED = cfg.META.STATIC_MAX_SPEED
STRAIGHT_MAX_LAT_DIST = cfg.META.STRAIGHT_MAX_LAT_DIST
STRAIGHT_MAX_HEADING_DIFF = cfg.META.STRAIGHT_MAX_HEADING_DIFF
UTURN_MIN_LON_DIST = cfg.META.UTURN_MIN_LON_DIST

# Length class params.
SHORT_MAX_DIST = cfg.META.SHORT_MAX_DIST
LONG_MIN_DIST = cfg.META.LONG_MIN_DIST
LONG_MAX_DIST = cfg.META.LONG_MAX_DIST


def track_dist_to_ego(tracks):
    """Track distance to ego.

    Args:
      tracks: ([pl.DataFrame]) full size object tracks.

    Returns:
      (np.array) track distance to ego, sized [N,].
    """
    H = cfg.TRACK.HISTORY_SIZE
    ego_xy = next(x[H, ["px", "py"]].to_numpy()
                  for x in tracks if x[0, "is_ego"] == 1)
    xys = [x[H, ["px", "py"]].to_numpy() for x in tracks]
    xys = np.concatenate(xys, axis=0)
    dists = np.linalg.norm(xys-ego_xy, ord=2, axis=-1)
    return dists


def track_length_class(tracks):
    """Classify track into 3 length types.


    Args:
      tracks: ([pl.DataFrame]) full size object tracks.

    Returns:
      ([str]) track length types, sized [N,].

    Track length class:
      0: "SHORT",
      1: "MEDIUM",
      2: "LONG",
      3: "INVALID"
    """
    xys = [x[["px", "py"]].to_numpy() for x in tracks]
    xys = np.stack(xys, axis=0)
    dists = get_polyline_length(xys)
    types = ["INVALID"] * len(tracks)
    for i, dist in enumerate(dists):
        if dist < SHORT_MAX_DIST:
            types[i] = "SHORT"
        elif dist > SHORT_MAX_DIST and dist < LONG_MIN_DIST:
            types[i] = "MEDIUM"
        elif dist > LONG_MIN_DIST and dist < LONG_MAX_DIST:
            types[i] = "LONG"
    return types


def track_orient_class(tracks):
    """Classify track into 8 orient types.

    Args:
      tracks: ([pl.DataFrame]) full size object tracks.

    Returns:
      ([str]) track orient types, sized [N,].

    Track orient class:
      0: "STATIONARY",
      1: "STRAIGHT",
      2: "STRAIGHT_LEFT",
      3: "STRAIGHT_RIGHT",
      4: "LEFT_U_TURN",
      5: "LEFT_TURN",
      6: "RIGHT_U_TURN",
      7: "RIGHT_TURN",
      8: "UNKNOWN",

    Reference: 
      https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/metrics/motion_metrics_utils.cc#L28

    Note:
      This is a bug in the original Waymo API implementation. State heading is in range [-PI,PI], the heading_diff is in range [-2PI,2PI].
      We can't directly use it for determining turning direction.
      e.g. start point heading is -170°, end point heading is 170°, the direction is straight, but the heading_diff=170°-(-170°)=340°!

      Instead we use vector angle diff with vector cross product to determine turning directions and degrees.
    """
    states = [x[["px", "py", "yaw", "speed"]].to_numpy() for x in tracks]
    states = np.stack(states, 0)  # [N,T,4]

    H = cfg.TRACK.HISTORY_SIZE
    diff = states[:, -1, :2] - states[:, H, :2]  # [N,2]

    # Final displacement.
    dist = np.linalg.norm(diff[:, :2], ord=2, axis=1)  # [N,]

    # Use vector angle and cross product to determine turning directions.
    start_yaw = states[:, H, 2]  # [N,]
    end_yaw = states[:, -1, 2]

    v0 = np.array([np.cos(start_yaw), np.sin(start_yaw)]).T  # [N,2]
    v1 = np.array([np.cos(end_yaw), np.sin(end_yaw)]).T  # [N,2]
    yaw_diff = vector_angle_diff(v0, v1)  # [N,], in range [0,PI]

    v0 = np.concatenate([v0, np.ones((len(v0), 1))], axis=1)  # [N,3]
    v1 = np.concatenate([v1, np.ones((len(v1), 1))], axis=1)  # [N,3]
    cross = np.cross(v0, v1)[:, 2]  # [N,]

    # Rotate.
    rot_xys = rotate(diff[:, None, :], -start_yaw)
    dx = rot_xys[:, 0, 0]
    dy = rot_xys[:, 0, 1]

    max_speed = np.maximum(states[:, H, 3], states[:, -1, 3])  # [N,]

    # The following logic is a little bit tricky, read reference for more details.
    types = [""] * len(tracks)
    for i in range(len(tracks)):
        if max_speed[i] < STATIC_MAX_SPEED and dist[i] < STATIC_MAX_DIST:
            types[i] = "STATIONARY"
            continue
        if dist[i] < STATIC_MAX_DIST:
            types[i] = "UNKNOWN"
            continue
        if yaw_diff[i] < STRAIGHT_MAX_HEADING_DIFF:
            if abs(dy[i]) < STRAIGHT_MAX_LAT_DIST:
                types[i] = "STRAIGHT"
                continue
            else:
                types[i] = "STRAIGHT_RIGHT" if dy[i] < 0 else "STRAIGHT_LEFT"
                continue
        if cross[i] < 0 and dy[i] < 0:
            types[i] = "RIGHT_U_TURN" if dx[i] < UTURN_MIN_LON_DIST else "RIGHT_TURN"
            continue
        if cross[i] > 0 and dy[i] > 0:
            types[i] = "LEFT_U_TURN" if dx[i] < UTURN_MIN_LON_DIST else "LEFT_TURN"
            continue
        types[i] = "UNKNOWN"
    return types


def track_padding(track, timestamps):
    """Padding track with specific timestamps.

    Args:
      track: (pl.DataFrame) agent track.
      timestamps: ([int]) input timestamps.

    Returns:
      (pl.DataFrame) padded tracks with timestamps.
    """
    if len(track) == len(timestamps):
        return track

    padded = timestamps.join(track, on="timestamp", how="outer").fill_null(0)
    return padded
