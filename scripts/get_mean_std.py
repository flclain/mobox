"""Calculate feature mean & std to normalize input features."""
import torch

from config.defaults import get_cfg
from mobox.datasets import construct_loader
from projects.wayformer.dataset import WaymoDataset

N = 200


def calc_mean_std():
    cfg = get_cfg()
    dataloader = construct_loader(cfg, mode="train")
    agents = []
    nearby = []
    map = []
    targets = []
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{N}")
        if batch_idx > N:
            break
        x = batch["agent"].tensor
        y = batch["nearby"].tensor
        z = batch["map"].tensor
        t = batch["target"].tensor
        print(x.max(), y.max(), z.max())
        print(x.min(), y.min(), z.min())
        agents.append(x)
        nearby.append(y)
        map.append(z)
        targets.append(t)

    agents = torch.cat(agents, 0)    # [N,T,1,D]
    nearby = torch.cat(nearby, 0)    # [N,T,S,D]
    map = torch.cat(map, 0)          # [N,T,S,D]
    targets = torch.cat(targets, 0)  # [N,T,D]

    agent_mean = agents.mean([0, 2])  # [T,D]
    agent_std = agents.std([0, 2])    # [T,D]

    nearby_mean = nearby.mean([0, 2])
    nearby_std = nearby.std([0, 2])

    map_mean = map.mean([0, 2])
    map_std = map.std([0, 2])

    target_mean = targets.mean([0])
    target_std = targets.std([0])

    mean_std = {
        "agent_mean": agent_mean,
        "agent_std": agent_std,
        "nearby_mean": nearby_mean,
        "nearby_std": nearby_std,
        "map_mean": map_mean,
        "map_std": map_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    torch.save(mean_std, "./cache/mean_std.pth")


if __name__ == "__main__":
    calc_mean_std()
