"""Extract scenario.pb from scenario tfrecord."""
import os
import glob
import tensorflow as tf

from tqdm import tqdm
from config.defaults import get_cfg
from mobox.data.protos import scenario_pb2


def extract(pb_file):
    record = tf.data.TFRecordDataset(pb_file)
    pb_dir = os.path.dirname(pb_file)
    pb_name = os.path.basename(pb_file)
    for i, x in enumerate(record):
        scenario_pb = scenario_pb2.Scenario()
        scenario_pb.ParseFromString(x.numpy())
        with open(f"{pb_dir}/{pb_name}_{i}.pb", "wb") as f:
            f.write(scenario_pb.SerializeToString())


if __name__ == "__main__":
    cfg = get_cfg()

    pb_files = glob.glob(f"{cfg.DATA.ROOT}/uncompressed/scenario/training/*", recursive=False)
    print(pb_files)
    for pb_file in tqdm(pb_files):
        extract(pb_file)
