import argparse
import copy
import os
import time
import warnings

import mmcv
import torch
from torchpack.utils.config import configs
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis.test import collect_results_cpu, collect_results_gpu
from mmdet3d.apis import single_gpu_test
from mmdet3d.apis.test import _stage2_records
from mmdet3d.apis.stage1_export import Stage1Exporter
from mmdet3d.apis.stage2_manifest import Stage2ManifestWriter, inject_stage2_corruption
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.models.fusion_models.stage2_missing_hook import install_stage2_missing_hook
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--export-level",
        choices=["none", "predictions", "features", "all"],
        default="none",
    )
    parser.add_argument("--export-sample-tokens", nargs="*", default=None)
    parser.add_argument("--export-every-n", type=int, default=1)
    parser.add_argument("--export-max-samples", type=int, default=None)
    parser.add_argument(
        "--export-dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--export-output-dir", default="outputs/stage1")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--corruption-type",
        choices=[
            "camera_lowlight",
            "lidar_dropout",
            "lidar_beam_reduction",
            "camera_extrinsic_translation",
            "camera_extrinsic_rotation",
            "camera_temporal_offset",
            "camera_missing",
            "lidar_missing",
        ],
    )
    parser.add_argument("--severity", type=int, choices=range(4), default=0)
    parser.add_argument("--corruption-seed", type=int, default=20260707)
    parser.add_argument("--affected-sensors", nargs="+", default=None)
    parser.add_argument(
        "--lowlight-candidate",
        choices=["legacy_candidate", "calibrated_candidate", "formal_candidate"],
        default="calibrated_candidate",
    )
    parser.add_argument("--temporal-history-file", default=None)
    parser.add_argument("--stage2-manifest-dir")
    parser.add_argument("--stage2-token-list-sha256")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    if args.export_every_n < 1:
        parser.error("--export-every-n must be at least 1")
    if args.corruption_type is None and args.severity:
        parser.error("--severity requires --corruption-type")
    return args


def get_bev_ranges(cfg):
    point_cloud_range = cfg.get("point_cloud_range")
    lidar_range = None
    if point_cloud_range:
        lidar_range = [
            point_cloud_range[0],
            point_cloud_range[1],
            point_cloud_range[3],
            point_cloud_range[4],
        ]
    camera_range = None
    camera = cfg.model.get("encoders", {}).get("camera")
    if camera:
        xbound = camera.vtransform.xbound
        ybound = camera.vtransform.ybound
        camera_range = [xbound[0], ybound[0], xbound[1], ybound[1]]
    fused_range = lidar_range or camera_range
    return {
        "camera_bev": camera_range,
        "lidar_bev": lidar_range,
        "prefusion": fused_range,
        "fused_bev": fused_range,
    }


def remove_unused_radar(cfg):
    if cfg.model.get("encoders", {}).get("radar"):
        return
    datasets = cfg.data.test if isinstance(cfg.data.test, list) else [cfg.data.test]
    for dataset in datasets:
        dataset.pipeline = [
            step for step in dataset.pipeline if step.type != "LoadRadarPointsMultiSweeps"
        ]
        for step in dataset.pipeline:
            if step.type == "Collect3D":
                step["keys"] = [key for key in step["keys"] if key != "radar"]


def multi_gpu_stage2_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Distributed test plus Stage 2 sample-record collection."""
    model.eval()
    results, records = [], []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        missing = getattr(model.module, "stage2_missing", None)
        for record in _stage2_records(data):
            if missing is not None:
                record["corruption_parameters"]["feature_branch_stats"] = {
                    **missing["hook"].last_stats,
                    "fused_abs_mean": missing.get("fused_abs_mean"),
                }
            records.append(record)
        if rank == 0:
            for _ in range(len(result) * world_size):
                prog_bar.update()
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    records = collect_results_cpu(records, len(dataset), None if tmpdir is None else tmpdir + "_records")
    return results, records


def main():
    args = parse_args()
    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher)

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(torch.cuda.current_device() if distributed else 0)

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    print(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.corruption_type is not None:
        inject_stage2_corruption(
            cfg,
            args.corruption_type,
            args.severity,
            args.corruption_seed,
            args.affected_sensors,
            args.lowlight_candidate,
            args.temporal_history_file,
        )
    remove_unused_radar(cfg)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get("encoders", {}).get("camera"):
        cfg.model.encoders.camera.backbone.init_cfg = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.corruption_type in ("camera_missing", "lidar_missing") and args.severity:
        install_stage2_missing_hook(model, args.corruption_type)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    exporter = None
    if args.export_level != "none":
        if distributed and args.export_level in ("features", "all"):
            raise ValueError("Distributed Stage 1 export supports predictions only")
        rank, _ = get_dist_info()
        if rank == 0:
            exporter = Stage1Exporter(
                model=model,
                dataset=dataset,
                output_dir=args.export_output_dir,
                export_level=args.export_level,
                config=args.config,
                checkpoint=args.checkpoint,
                seed=args.seed,
                sample_tokens=args.export_sample_tokens,
                every_n=args.export_every_n,
                max_samples=args.export_max_samples,
                dtype=args.export_dtype,
                run_id=args.run_id,
                bev_ranges=get_bev_ranges(cfg),
            )

    stage2_writer = None
    if args.stage2_manifest_dir:
        if args.corruption_type is None:
            raise ValueError("Stage 2 manifests require corruption inference")
        rank, _ = get_dist_info()
        if rank == 0:
            stage2_writer = Stage2ManifestWriter(
                args.stage2_manifest_dir,
                args.run_id,
                config=args.config,
                checkpoint=args.checkpoint,
                dataset_version=dataset.version,
                token_list_sha256=args.stage2_token_list_sha256,
            )

    try:
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(
                model,
                data_loader,
                exporter=exporter,
                stage2_writer=stage2_writer,
            )
        else:
            model = MMDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
            )
            if stage2_writer is None:
                outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
            else:
                outputs, records = multi_gpu_stage2_test(
                    model, data_loader, args.tmpdir, args.gpu_collect
                )
                for record in records or []:
                    stage2_writer.add(record)
            if exporter is not None:
                exporter.write_batch(outputs)
    except Exception:
        if exporter is not None:
            exporter.close("failed")
        if stage2_writer is not None:
            stage2_writer.close("failed")
        raise

    rank, _ = get_dist_info()
    metrics = None
    try:
        if rank == 0:
            if args.out:
                print(f"\nwriting results to {args.out}")
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                eval_kwargs = cfg.get("evaluation", {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                    "interval",
                    "tmpdir",
                    "start",
                    "gpu_collect",
                    "save_best",
                    "rule",
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                metrics = dataset.evaluate(outputs, **eval_kwargs)
                print(metrics)
    except Exception:
        if exporter is not None:
            exporter.close("failed")
        if stage2_writer is not None:
            stage2_writer.close("failed")
        raise
    else:
        if exporter is not None:
            exporter.close(metrics=metrics)
        if stage2_writer is not None:
            stage2_writer.close(metrics=metrics)


if __name__ == "__main__":
    main()
