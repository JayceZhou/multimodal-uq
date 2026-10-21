import argparse
import json
from pathlib import Path

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.model.encoders.camera.backbone.init_cfg = None

    dataset_cfg = cfg.data.train
    inner = dataset_cfg.dataset if dataset_cfg.type == "CBGSDataset" else dataset_cfg
    inner.ann_file = args.ann_file
    inner.pipeline = [
        step
        for step in inner.pipeline
        if step.type not in ("ObjectPaste", "LoadRadarPointsMultiSweeps")
    ]
    for step in inner.pipeline:
        if step.type == "Collect3D":
            step["keys"] = [key for key in step["keys"] if key != "radar"]

    # A tiny smoke subset can omit classes, which makes CBGS divide by zero.
    dataset = build_dataset(inner)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    model = build_model(cfg.model)
    load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    assert trainable and all(name.startswith("heads.aux_object.") for name, _ in trainable)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=cfg.optimizer.lr)
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {
        id(parameter) for _, parameter in trainable
    }

    frozen_before = {
        name: parameter.detach().cpu().clone() for name, parameter in frozen
    }
    trainable_before = {
        name: parameter.detach().cpu().clone() for name, parameter in trainable
    }
    model = model.cuda()
    parallel = MMDataParallel(model, device_ids=[0])
    parallel.train()
    iterator = iter(data_loader)
    losses = []
    for _ in range(args.iterations):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        output = parallel(return_loss=True, **batch)
        loss, _ = model._parse_losses(output)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(parameter.grad is None for _, parameter in frozen)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    frozen_changed = [
        name for name, parameter in frozen if not torch.equal(frozen_before[name], parameter.detach().cpu())
    ]
    trainable_changed = [
        name for name, parameter in trainable if not torch.equal(trainable_before[name], parameter.detach().cpu())
    ]
    assert not frozen_changed
    assert trainable_changed

    total_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    report = {
        "iterations": args.iterations,
        "losses": losses,
        "trainable_parameters": [name for name, _ in trainable],
        "trainable_count": trainable_count,
        "total_count": total_count,
        "trainable_ratio": trainable_count / total_count,
        "trainable_changed": trainable_changed,
        "frozen_changed": frozen_changed,
        "frozen_gradients_are_none": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
