import mmcv
import torch


def _stage2_records(data):
    container = data["metas"]
    if isinstance(container, (list, tuple)):
        container = container[0]
    metas = container.data
    while isinstance(metas, (list, tuple)) and len(metas) == 1:
        metas = metas[0]
    metas = [metas] if isinstance(metas, dict) else metas
    return [meta["stage2_manifest"] for meta in metas if "stage2_manifest" in meta]


def single_gpu_test(model, data_loader, exporter=None, stage2_writer=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        if exporter is not None:
            exporter.write_batch(result)
        if stage2_writer is not None:
            missing = getattr(model.module, "stage2_missing", None)
            for record in _stage2_records(data):
                if missing is not None:
                    record["corruption_parameters"]["feature_branch_stats"] = {
                        **missing["hook"].last_stats,
                        "fused_abs_mean": missing.get("fused_abs_mean"),
                    }
                stage2_writer.add(record)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
