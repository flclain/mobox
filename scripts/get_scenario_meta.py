"""
Generate scenario meta info.

Scenario meta is a DataFrame contains:
------------------------------------------------------------------------------
scenario_file | track_id | track_orient | track_length | is_vru | dist_to_ego
------------------------------------------------------------------------------
"""
import glob
import argparse
import polars as pl

from tqdm import tqdm
from multiprocessing import Pool
from collections import namedtuple

from mobox.utils.scenario import parse_scenario_from_pb
from mobox.utils.track import track_orient_class, track_length_class, track_dist_to_ego


META_FIELD_NAMES = ["scenario_file", "track_id",
                    "track_orient", "track_length", "is_vru", "dist_to_ego"]
MetaRow = namedtuple(typename="MetaRow", field_names=META_FIELD_NAMES)


def generate_scenario_meta(scenario, pb_file):
    orient_cls = track_orient_class(scenario.focused_tracks)
    length_cls = track_length_class(scenario.focused_tracks)
    dist_to_ego = track_dist_to_ego(scenario.focused_tracks)
    rows = []
    for i, track in enumerate(scenario.focused_tracks):
        row = MetaRow(
            scenario_file=pb_file,
            track_id=track[0, "track_id"],
            is_vru=track[0, "is_vru"],
            track_orient=orient_cls[i],
            track_length=length_cls[i],
            dist_to_ego=dist_to_ego[i],
        )
        rows.append(row)
    return rows


def process_once(pb_file):
    scenario = parse_scenario_from_pb(pb_file)
    if not scenario:
        return []
    row = generate_scenario_meta(scenario, pb_file)
    return row


def main(args):
    tasks = []
    for pb_file in glob.glob(f"{args.pb_dir}/**/*.pb", recursive=True):
        tasks.append(pb_file)

    rows = []
    with Pool(8) as pool:
        ret = tqdm(pool.imap_unordered(process_once, tasks), total=len(tasks))
        rows = sum(ret, [])

    df = pl.from_records(rows, columns=MetaRow._fields)
    df.write_csv("./csv/meta.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate scenario meta.")

    parser.add_argument("pb_dir", help="scenario pb directory")
    args = parser.parse_args()

    main(args)
