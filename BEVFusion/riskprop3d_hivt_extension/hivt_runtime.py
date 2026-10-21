"""Native HiVT graph contract, supervised target loss and prediction coordinates."""
import os
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "python"))
sys.path.insert(0, str(HERE / "third_party/MapUncertaintyPrediction/HiVT_modified"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hivt_mpl_cache")
import numpy as np
import torch
from torch_geometric.data import Batch
from torch.utils.data import Dataset
from models.hivt import HiVT
from utils import TemporalData

MODEL = dict(historical_steps=5, future_steps=6, num_modes=6, rotate=True,
             node_dim=2, edge_dim=2, embed_dim=128, num_heads=8, dropout=.2,
             num_temporal_layers=4, num_global_layers=3, local_radius=50,
             parallel=False, lr=5e-4, weight_decay=1e-4, T_max=64, method="base")


def graph(positions, valid, lane, origin, row):
    positions = torch.as_tensor(np.asarray(positions), dtype=torch.float32)
    valid = torch.as_tensor(np.asarray(valid), dtype=torch.bool)
    lane = torch.as_tensor(np.asarray(lane), dtype=torch.float32)
    count = len(positions)
    x = torch.zeros_like(positions)
    x[:, 1:] = torch.where((valid[:, 1:] & valid[:, :-1])[..., None], positions[:, 1:] - positions[:, :-1], 0.)
    bos = torch.cat((valid[:, :1], ~valid[:, :-1] & valid[:, 1:]), axis=1)
    padding = torch.cat((~valid, torch.ones((count, 6), dtype=torch.bool)), axis=1)
    nodes = torch.arange(count)
    edge = torch.stack((nodes.repeat_interleave(count), nodes.repeat(count)))
    edge = edge[:, edge[0] != edge[1]]
    lane_edge = torch.stack((torch.arange(len(lane)).repeat_interleave(count), nodes.repeat(len(lane))))
    angle = torch.atan2(x[:, -1, 1], x[:, -1, 0])
    data = TemporalData(x=x, positions=positions, edge_index=edge, num_nodes=count, y=None,
                        padding_mask=padding, bos_mask=bos, rotate_angles=angle,
                        lane_vectors=lane, lane_actor_index=lane_edge,
                        lane_actor_vectors=lane[lane_edge[0]] - positions[lane_edge[1], -1])
    data.target_index = torch.tensor([0], dtype=torch.long)
    data.origin = torch.tensor(np.asarray(origin)[None], dtype=torch.float32)
    data.row = torch.tensor([row], dtype=torch.long)
    return data


class ContextDataset(Dataset):
    def __init__(self, condition, labels=False):
        directory = HERE / "data_v1" / condition
        manifest = json.loads((directory / "manifest.json").read_text())
        for filename, key in (("geometry.npz", "geometry_sha256"), ("labels.npz", "labels_sha256")):
            digest = hashlib.sha256()
            with (directory / filename).open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != manifest[key]:
                raise ValueError("Context input hash changed: " + str(directory / filename))
        with np.load(directory / "geometry.npz", allow_pickle=False) as arrays:
            self.arrays = {k: arrays[k] for k in arrays.files}
        self.future = None
        if labels:
            with np.load(directory / "labels.npz", allow_pickle=False) as arrays:
                self.future = arrays["future"]
        assert self.arrays["target_xy"].shape[1:] == (5, 2)
        assert self.arrays["target_mask"].shape == (len(self), 5)
        assert self.arrays["target_mask"][:, -3:].all()
        assert len(self) == manifest["targets"]
        if self.future is not None:
            assert self.future.shape == (len(self), 6, 2)

    def __len__(self):
        return len(self.arrays["target_xy"])

    def __getitem__(self, row):
        a = self.arrays
        neighbors = a["neighbor_index"][row]
        neighbors = neighbors[neighbors >= 0]
        origin = a["target_xy"][row, -1]
        positions = np.concatenate((a["target_xy"][row:row + 1], a["agent_xy"][neighbors])) - origin
        valid = np.concatenate((a["target_mask"][row:row + 1], a["agent_mask"][neighbors]))
        offsets = a["lane_row_offsets"]
        indices = a["lane_index"][offsets[row]:offsets[row + 1]]
        lane = a["lane_xy"][indices] - origin
        data = graph(positions, valid, lane, origin, row)
        if self.future is not None:
            data.label = torch.tensor(self.future[row:row + 1], dtype=torch.float32)
        return data


def collate(rows):
    return Batch.from_data_list(rows)


def perturb(data, current):
    """Write the LiDAR XY draw through every field that reads current target XY."""
    data = data.clone()
    assert data.y is None
    if not torch.isfinite(current).all():
        raise FloatingPointError("Non-finite current XY tensor; no clipping or redraw")
    target = data.target_index
    data.positions[target, -1] = current - data.origin
    data.x[target, -1] = data.positions[target, -1] - data.positions[target, -2]
    data.rotate_angles[target] = torch.atan2(data.x[target, -1, 1], data.x[target, -1, 0])
    edge = data.lane_actor_index
    data.lane_actor_vectors = data.lane_vectors[edge[0]] - data.positions[edge[1], -1]
    return data


def forward(model, data):
    assert data.y is None
    features = data.clone()
    if "label" in features:
        del features.label
    prediction, logits = model(features)
    target = data.target_index
    value = prediction[:, target].permute(1, 0, 2, 3)
    probability = torch.softmax(logits[target], dim=-1)
    angle = data.rotate_angles[target]
    c, s = torch.cos(angle), torch.sin(angle)
    inverse = torch.stack((c, s, -s, c), axis=-1).reshape(-1, 2, 2)
    means = torch.matmul(value[..., :2], inverse[:, None]) + (data.positions[target, -1] + data.origin)[:, None, None]
    scales = value[..., 2:]
    if not all(torch.isfinite(v).all() for v in (means, scales, logits, prediction, probability)) or not torch.all(scales > 0):
        raise FloatingPointError("Non-finite native prediction or nonpositive Laplace scale")
    return means, scales, probability, inverse, value[..., :2]


def loss(model, data):
    # Native regression/classification formula, with labels isolated from forward.
    assert data.y is None
    features = data.clone()
    del features.label
    prediction, logits = model(features)
    target = data.target_index
    angle = data.rotate_angles[target]
    c, s = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack((c, -s, s, c), axis=-1).reshape(-1, 2, 2)
    label = torch.bmm(data.label - (data.positions[target, -1] + data.origin)[:, None], rotation)
    value = prediction[:, target]
    distance = torch.linalg.vector_norm(value[..., :2] - label, dim=-1).sum(-1)
    best = distance.argmin(0)
    regression = model.reg_loss(value[best, torch.arange(len(target), device=target.device)], label)
    classification = model.cls_loss(logits[target], torch.softmax(-distance / 6, dim=0).t().detach())
    result = regression + classification
    assert torch.isfinite(result)
    return result


def self_check():
    xy = np.asarray([[[t, 0] for t in range(5)], [[t, 3] for t in range(5)]], dtype=np.float32)
    valid = np.ones((2, 5), dtype=bool)
    valid[1, :2] = False
    lane = np.asarray([[0, 1], [6, 1]], dtype=np.float32)
    first = graph(xy, valid, lane, [10, 20], 0)
    batch = collate([first, first])
    assert batch.target_index.tolist() == [0, 2]
    assert batch.lane_actor_index.max(1).values.tolist() == [3, 3]
    assert not batch.x[1, :3].any() and batch.bos_mask[1, 2]
    changed = perturb(batch, torch.tensor([[17., 23.], [18., 24.]]))
    assert torch.equal(changed.positions[batch.target_index, -1], torch.tensor([[7., 3.], [8., 4.]]))
    assert torch.equal(batch.positions, torch.cat((first.positions, first.positions)))
    assert torch.equal(changed.positions[[1, 3]], batch.positions[[1, 3]])
    assert torch.equal(changed.lane_vectors, batch.lane_vectors)
    assert changed.y is None and batch.y is None
    print("PASS: PyG batch indices, masked displacement, current draw propagation and fixed context")


if __name__ == "__main__":
    self_check()
