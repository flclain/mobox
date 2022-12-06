import math
import polars as pl

from collections import namedtuple
from mobox.base.protos import scenario_pb2
from mobox.base.schema import Scenario, VRUType

MAP_FIELD_NAMES = ["id", "px", "py", "type", "sub_type"]
AGENT_FIELD_NAMES = ["timestamp", "px", "py", "yaw", "vx", "vy", "speed", "length",
                     "width", "object_type", "track_id", "is_vru", "is_valid", "is_ego", "is_focused"]

MapRow = namedtuple(typename="MapRow", field_names=MAP_FIELD_NAMES)
AgentRow = namedtuple(typename="AgentRow", field_names=AGENT_FIELD_NAMES)


def parse_map(scenario_pb):
    rows = []
    for x in scenario_pb.map_features:
        type = x.WhichOneof("feature_data")
        if type not in ["lane", "road_edge", "road_line"]:
            continue

        polyline = getattr(x, type).polyline
        for point in polyline:
            row = MapRow(
                id=x.id,
                type=type,
                sub_type=getattr(x, type).type,
                px=point.x,
                py=point.y,
            )
            rows.append(row)
    df = pl.from_records(rows, columns=MapRow._fields)
    return df


def parse_tracks(scenario_pb):
    ego_idx = scenario_pb.sdc_track_index
    timestamps = [int(1000*x) for x in list(scenario_pb.timestamps_seconds)]  # 10+1+80
    tracks_to_predict = [x.track_index for x in scenario_pb.tracks_to_predict]

    rows = []
    for i, track in enumerate(scenario_pb.tracks):
        is_ego = (i == ego_idx)
        is_vru = (track.object_type in VRUType)
        last_x, last_y = track.states[0].center_x, track.states[0].center_y
        for ts, state in zip(timestamps, track.states):
            diff = math.hypot(state.center_x - last_x, state.center_y - last_y)
            last_x, last_y = state.center_x, state.center_y
            row = AgentRow(
                timestamp=ts,
                px=state.center_x,
                py=state.center_y,
                vx=state.velocity_x,
                vy=state.velocity_y,
                speed=math.hypot(state.velocity_x, state.velocity_y),
                yaw=state.heading,
                length=state.length,
                width=state.width,
                object_type=track.object_type,
                track_id=(-1 if is_ego else track.id),
                is_valid=(state.valid and diff < 5),  # 50m/s = 180km/h
                is_ego=is_ego,
                is_vru=is_vru,
                is_focused=(i in tracks_to_predict and not is_vru) or is_ego,
            )
            rows.append(row)
    df = pl.from_records(rows, columns=AgentRow._fields)
    return df


def parse_scenario_from_pb(pb_file):
    scenario_pb = scenario_pb2.Scenario()
    with open(pb_file, "rb") as f:
        scenario_pb.ParseFromString(f.read())
    map = parse_map(scenario_pb)
    tracks = parse_tracks(scenario_pb)
    scenario = Scenario()
    scenario.scenario_id = scenario_pb.scenario_id
    scenario.timestamps = [int(1000*x) for x in list(scenario_pb.timestamps_seconds)]
    scenario.tracks = tracks.partition_by("track_id")
    scenario.map = map
    return scenario
