import torch
import numpy as np
import polars as pl
import torch.utils.data as data

from torchbox.utils.misc import NestedTensor
from torchbox.datasets import DATASET_REGISTRY
from torchbox.utils.decorators import valid_return

from projects.wayformer.feature_generator import WaymoFeatureGenerator
from projects.wayformer.scenario_generator import WaymoScenarioGenerator


@DATASET_REGISTRY.register()
class WaymoDataset(data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.train = (mode == "train")
        self.feat_gen = WaymoFeatureGenerator(cfg)
        self.scen_gen = WaymoScenarioGenerator(cfg)
        self.N = len(self.scen_gen)
        # self.mean_std = torch.load("./cache/mean_std.pth")

    @valid_return
    def __getitem__(self, idx):
        scenario, track_id = self.scen_gen.get()
        agent_feats = self.feat_gen.agent_feats(scenario, track_id)
        nearby_feats = self.feat_gen.nearby_agent_feats(scenario, track_id)
        map_feats = self.feat_gen.roadgraph(scenario, track_id)
        if agent_feats is None or nearby_feats is None or map_feats is None:
            return None
        return {"agent": agent_feats, "nearby": nearby_feats, "map": map_feats}

    # def zero_mean(self, batch):
    #     H = self.cfg.TRACK.HISTORY_SIZE + 1
    #     agent, nearby, map = batch["agent"], batch["nearby"], batch["map"]
    #
    #     agent = batch["agent"]
    #     mean = self.mean_std["agent_mean"][:H, None, :]
    #     std = self.mean_std["agent_std"][:H, None, :].clamp(min=1)
    #     agent.tensor = (agent.tensor - mean) / std
    #
    #     mean = self.mean_std["nearby_mean"][:H, None, :]
    #     std = self.mean_std["nearby_std"][:H, None, :].clamp(min=1)
    #     nearby.tensor = (nearby.tensor - mean) / std
    #
    #     mean = self.mean_std["map_mean"][:, None, :]
    #     std = self.mean_std["map_std"][:, None, :].clamp(min=1)
    #     map.tensor = (map.tensor - mean) / std
    #     return batch
    #
    # def backup_location(self, batch):
    #     x_agent = batch["agent"].tensor[:, -1, :, :2]     # [N,1,2]
    #     x_nearby = batch["nearby"].tensor[:, -1, :, :2]   # [N,S,2]
    #     x_map = batch["map"].tensor[:, 0, :, :2]          # [N,S,2]
    #     x = torch.cat([x_agent, x_nearby, x_map], dim=1)  # [N,L,2]
    #     batch["loc"] = x
    #     return batch

    def collate_fn(self, batch):
        H = self.cfg.TRACK.HISTORY_SIZE
        agent = [torch.from_numpy(x["agent"]) for x in batch]
        nearby = [torch.from_numpy(x["nearby"]) for x in batch]
        map = [torch.from_numpy(x["map"]) for x in batch]

        agent = torch.stack(agent, 0)
        nearby = torch.stack(nearby, 0)
        map = torch.stack(map, 0)

        target = NestedTensor(agent[:, H:, 0, :2], agent[:, H:, 0, -1])
        agent = NestedTensor(agent[:, :H, :, :-1], agent[:, :H, :, -1])
        nearby_target = NestedTensor(nearby[:, H:, :, :2], nearby[:, H:, :, -1])
        nearby = NestedTensor(nearby[:, :H, :, :-1], nearby[:, :H, :, -1])
        map = NestedTensor(map[..., :-1], map[..., -1])

        batch = {"agent": agent, "nearby": nearby, "map": map,
                 "target": target, "nearby_target": nearby_target}
        return batch

    def __len__(self):
        return 10000


def test_dataset():
    from config.defaults import get_cfg
    from torchbox.datasets import build_dataset, DATASET_REGISTRY
    cfg = get_cfg()
    dataset = build_dataset("WaymoDataset", cfg, "train")
    print(dataset[0])
    print(dataset[1])
    print(len(dataset))


def test_dataloader():
    from config.defaults import get_cfg
    from torchbox.datasets import construct_loader
    cfg = get_cfg()
    dataloader = construct_loader(cfg, mode="train")
    for batch_idx, batch in enumerate(dataloader):
        print(batch_idx)
        print(batch["agent"].tensor.shape)
        print(batch["nearby"].tensor.shape)
        print(batch["map"].tensor.shape)
        print(batch["target"].tensor.shape)
        print(batch["nearby_target"].tensor.shape)
        break


if __name__ == "__main__":
    # Need to change multiprocessing start method from "fork" to "spawn",
    # otherwise polars groupby will freeze.
    # torch.multiprocessing.set_start_method("spawn")
    # test_dataset()
    test_dataloader()
