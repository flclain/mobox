import glob
import math
import polars as pl
import tensorflow as tf

from collections import namedtuple
from mobox.data.protos import scenario_pb2
from mobox.data.schema import Scenario, VRUType


MAP_FIELD_NAMES = ["id", "px", "py", "type", "sub_type"]
AGENT_FIELD_NAMES = ["timestamp", "px", "py", "yaw", "vx", "vy", "speed",
                     "length", "width", "object_type", "track_id", "is_vru", "is_valid", "is_ego", "is_focused"]

MapRow = namedtuple(typename="MapRow", field_names=MAP_FIELD_NAMES)
AgentRow = namedtuple(typename="AgentRow", field_names=AGENT_FIELD_NAMES)


class WaymoScenarioGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pb_files = glob.glob(
            f"{self.cfg.DATA.ROOT}/uncompressed/scenario/training/*", recursive=False)

    def parse_lanes(self, scenario_pb):
        rows = []
        for x in scenario_pb.map_features:
            if not x.WhichOneof("feature_data") == "lane":
                continue
            for point in x.lane.polyline:
                row = MapRow(
                    id=x.id,
                    type="lane",
                    sub_type=x.lane.type,
                    px=point.x,
                    py=point.y,
                )
                rows.append(row)
        df = pl.from_records(rows, columns=MapRow._fields)
        return df

    def parse_boundaries(self, scenario_pb):
        rows = []
        for x in scenario_pb.map_features:
            if not x.WhichOneof("feature_data") == "road_edge":
                continue
            for point in x.road_edge.polyline:
                row = MapRow(
                    id=x.id,
                    type="road_edge",
                    sub_type=x.road_edge.type,
                    px=point.x,
                    py=point.y,
                )
                rows.append(row)
        df = pl.from_records(rows, columns=MapRow._fields)
        return df

    def parse_tracks(self, scenario_pb):
        ego_idx = scenario_pb.sdc_track_index
        timestamps = [int(1000*x) for x in list(scenario_pb.timestamps_seconds)]  # 10+1+80
        tracks_to_predict = [x.track_index for x in scenario_pb.tracks_to_predict]

        rows = []
        for i, track in enumerate(scenario_pb.tracks):
            for ts, state in zip(timestamps, track.states):
                row = AgentRow(
                    timestamp=ts,
                    px=state.center_x,
                    py=state.center_y,
                    vx=state.velocity_x,
                    vy=state.velocity_y,
                    speed=math.sqrt(state.velocity_x**2 + state.velocity_y**2),
                    yaw=state.heading,
                    length=state.length,
                    width=state.width,
                    object_type=track.object_type,
                    track_id=track.id,
                    is_valid=state.valid,
                    is_ego=(i == ego_idx),
                    is_vru=(track.object_type in VRUType),
                    is_focused=(i in tracks_to_predict),
                )
                rows.append(row)
        df = pl.from_records(rows, columns=AgentRow._fields)
        return df

    def parse_scenario_from_pb(self, scenario_pb):
        lanes = self.parse_lanes(scenario_pb)
        boundaries = self.parse_boundaries(scenario_pb)
        tracks = self.parse_tracks(scenario_pb)
        scenario = Scenario()
        scenario.scenario_id = scenario_pb.scenario_id
        scenario.timestamps = [int(1000*x) for x in list(scenario_pb.timestamps_seconds)]
        scenario.tracks = tracks.partition_by("track_id")
        scenario.map = pl.concat([lanes, boundaries])
        return scenario

    @property
    def scenarios(self):
        for pb_file in self.pb_files:
            record = tf.data.TFRecordDataset(pb_file)
            for x in record:
                scenario_pb = scenario_pb2.Scenario()
                scenario_pb.ParseFromString(x.numpy())
                scenario = self.parse_scenario_from_pb(scenario_pb)
                yield scenario

    def __len__(self):
        return 100


if __name__ == "__main__":
    from config.defaults import get_cfg
    cfg = get_cfg()
    gen = WaymoScenarioGenerator(cfg)
    for scenario in gen.scenarios:
        print(scenario)
        print(scenario.focused_tracks)
        break
