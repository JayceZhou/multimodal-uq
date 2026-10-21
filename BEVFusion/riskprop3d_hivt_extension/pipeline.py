"""Supervise independent recovery, context preparation and native host stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / "pipeline_v1"
GPUS = (0, 2, 3)


def status(name, **values):
    OUT.mkdir(exist_ok=True)
    target = OUT / (name + ".json")
    tmp = target.with_suffix(".partial")
    tmp.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    tmp.replace(target)


def wait_for(paths):
    paths = list(paths)
    while not all(p.exists() for p in paths):
        for failure in (HERE / "recovery_v1/failure.json", OUT / "cpu_failure.json", OUT / "gpu_failure.json"):
            if failure.exists():
                raise RuntimeError(failure.read_text())
        time.sleep(5)


def execute(name, arguments, physical=None):
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    if physical is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(physical)
    log = (OUT / "logs" / (name + ".log")).open("x")
    process = subprocess.Popen([sys.executable, *arguments], cwd=HERE.parent, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    return process, log


def single(name, arguments, physical=None):
    process, log = execute(name, arguments, physical)
    code = process.wait()
    log.close()
    if code:
        raise RuntimeError("Task failed: {} exit {}; see logs".format(name, code))


def available():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    values = {int(line.split(",")[0]): tuple(int(v) for v in line.split(",")[1:]) for line in result.splitlines()}
    return {gpu for gpu in GPUS if values[gpu][0] < 1000 and values[gpu][1] < 10}, result


def parallel(tasks, stage):
    pending, active, completed = list(tasks), {}, []
    while pending or active:
        free, inspection = available()
        for gpu in GPUS:
            if gpu not in active and gpu in free and pending:
                name, arguments = pending.pop(0)
                process, log = execute(name, arguments + ["--gpu", str(gpu)], gpu)
                active[gpu] = (process, log, name)
                print(json.dumps({"stage": stage, "task": name, "physical_gpu": gpu, "pid": process.pid,
                                  "gpu_inspection": inspection}), flush=True)
        failed = []
        for gpu, (process, log, name) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                record = {"task": name, "gpu": gpu, "exit_code": code}
                (failed if code else completed).append(record)
                del active[gpu]
        status("gpu_status", stage=stage, active={str(g): {"task": v[2], "pid": v[0].pid} for g, v in active.items()},
               pending=len(pending), completed=completed, failed=failed)
        if failed:
            raise RuntimeError("GPU stage failed: " + json.dumps(failed))
        time.sleep(5)


def prepare_cpu():
    for condition, split, shards in (("train", "train", 6), ("label_cal", "label_cal", 1), ("clean_s0", "dev", 1)):
        wait_for(HERE / "recovery_v1/raw" / ("{}_{:02d}".format(split, i)) / "accepted.json" for i in range(shards))
        status("cpu_status", stage="build_" + condition)
        single("build_" + condition, [str(HERE / "build_context.py"), condition])
    status("cpu_status", stage="CLEAN_CONTEXT_COMPLETE")


def hosts():
    wait_for([HERE / "recovery_v1/recovery_manifest.json"])
    tasks = [("background_{}_{:02d}".format(condition, shard),
              [str(HERE / "context_export.py"), "export", "--condition", condition, "--shard", str(shard)])
             for condition, shards in (("dropout_s3", 2), ("rotation_s3", 2), ("official_clean", 4)) for shard in range(shards)]
    parallel(tasks, "perception_background_recovery")
    for condition in ("dropout_s3", "rotation_s3", "official_clean"):
        single("build_" + condition, [str(HERE / "build_context.py"), condition])
    wait_for([HERE / "data_v1/train/manifest.json", HERE / "data_v1/label_cal/manifest.json", HERE / "data_v1/clean_s0/manifest.json"])
    parallel([("engineering", [str(HERE / "run_host.py"), "engineering"])], "real_context_engineering")
    single("host_freeze", [str(HERE / "run_host.py"), "freeze"])
    parallel([("host_train_" + str(i), [str(HERE / "run_host.py"), "train", "--replicate", str(i)]) for i in range(3)], "three_host_training")
    parallel([("host_calibrate_" + str(i), [str(HERE / "run_host.py"), "calibrate", "--replicate", str(i)]) for i in range(3)], "clean_output_calibration")
    status("gpu_status", stage="HOSTS_READY_FOR_EVALUATION", task_incomplete=True)


def finish():
    wait_for(HERE / "host_v1" / ("run_" + str(i)) / "calibration.json" for i in range(3))
    single("evaluation_self_check", [str(HERE / "evaluate.py"), "self-check"])
    single("statistics_self_check", [str(HERE / "summarize.py"), "--self-check"])
    parallel([("support_probe_" + str(i), [str(HERE / "evaluate.py"), "probe", "--replicate", str(i)]) for i in range(3)], "stochastic_support_probe")
    single("evaluation_freeze", [str(HERE / "evaluate.py"), "freeze"])
    parallel([("main_evaluate_" + str(i), [str(HERE / "evaluate.py"), "evaluate", "--replicate", str(i)]) for i in range(3)], "four_laws_four_populations")
    parallel([("sensitivity_" + str(i), [str(HERE / "evaluate.py"), "sensitivity", "--replicate", str(i)]) for i in range(3)], "MC_sensitivity")
    parallel([("controls_" + str(i), [str(HERE / "evaluate_controls.py"), "--replicate", str(i)]) for i in range(3)], "CV_MLP_same_targets")
    single("final_statistics", [str(HERE / "summarize.py")])
    status("finish_status", stage="LEVEL_I_COMPLETE", report="REPORT.md", task_incomplete=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-cpu", "hosts", "finish"))
    args = parser.parse_args()
    try:
        {"prepare-cpu": prepare_cpu, "hosts": hosts, "finish": finish}[args.command]()
    except Exception as error:
        status("cpu_failure" if args.command == "prepare-cpu" else "gpu_failure",
               error=type(error).__name__, message=str(error), time_unix=time.time())
        raise
