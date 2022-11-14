import polars as pl

from typing import List, Optional

ObjectType = {
    0: "UNSET",
    1: "VEHICLE",
    2: "PEDESTRAIN",
    3: "CYCLIST",
    4: "OTHER",
}

VRUType = (2, 3)  # PEDESTRAIN, CYCLIST

# Reference:
#   https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/metrics/motion_metrics_utils.h#L29
TrackType = {
    0: "STATIONARY",
    1: "STRAIGHT",
    2: "STRAIGHT_LEFT",
    3: "STRAIGHT_RIGHT",
    4: "LEFT_U_TURN",
    5: "LEFT_TURN",
    6: "RIGHT_U_TURN",
    7: "RIGHT_TURN",
    8: "UNKNOWN",  # do not use UNKNOWN for training
}


class Scenario:
    """Bundles all data associated with a scenario.

    Args:
      scenario_id: (int) unique ID associated with this scenario.
      timestamps: ([int]) all timestamps.
      tracks: ([pl.DataFrame]) all tracks.
      focus_track_ids: ([int]) object track ids with full length.
      map: (pl.DataFrame) map data.

    Reference: 
      https://github.com/argoai/av2-api/blob/main/src/av2/datasets/motion_forecasting/data_schema.py#L101

    TODO:
      Save as pb file.
    """
    pb: str
    scenario_id: int
    timestamps: List[int]
    tracks: List[pl.DataFrame]
    map: Optional[pl.DataFrame]

    def query_track_by_id(self, track_id):
        for track in self.tracks:
            if track[0, "track_id"] == track_id:
                return track
        raise Exception(f"Not found track_id {track_id} in scenario.")

    @property
    def focused_tracks(self):
        """Return full length tracks."""
        # N = len(self.timestamps)
        # return [x for x in self.tracks if x.filter(pl.col("is_focused") == 1).shape[0] == N]
        return [x for x in self.tracks if x[0, "is_focused"] == 1]

    def __str__(self):
        return f"[SCENARIO id: {self.scenario_id} track_len: {len(self.timestamps)} num_agents: {len(self.tracks)}]"

    def __repr__(self):
        return self.__str__()
