# Preservation runtime protocol

## Frozen timing evidence

The retained timing endpoint is the clean development population: 34,856 targets, processed in 137 timed chunks per replicate with at most 256 targets per chunk. Three paired replicates were recorded. The reported `38.55x` value is

```text
mean_r(teacher_total_seconds_r / student_total_seconds_r)
= 38.551418913714x.
```

It is not the ratio of two rounded means. Per-replicate totals and derived per-target latencies are in `runtime_measurements.csv`.

## Timed regions

- **Teacher (`D1_mc_teacher`)**: synchronize CUDA; start `time.perf_counter`; construct NumPy random state draws; sample 64 HTC current states; replace the target current state; execute the frozen MLP on all 64 states; construct the calibrated MLP process noise; synchronize CUDA; stop the clock.
- **Student (`D3_risk_distilled`)**: synchronize CUDA; start `time.perf_counter`; construct the raw probability token; construct and transfer motion/token tensors to the device; execute one student forward pass under `torch.inference_mode`; copy the output arrays to CPU as part of `numpy_output`; synchronize CUDA; stop the clock.
- Downstream mixture sampling, trajectory metrics, teacher-fidelity metrics, and bootstrap inference are excluded from both timed regions.
- The upstream probabilistic-head forward pass is excluded from both methods; timing starts from retained HTC parameters.

## Warm-up and repetition

Before the first timed clean chunk in each replicate, the evaluator runs one unrecorded forward pass for each learned student and one unrecorded teacher chunk. Formal timing then covers every one of the 137 clean evaluation chunks once. These are full-population accumulated timings, not repeated timing of one synthetic batch. CUDA synchronization occurs immediately before and after every recorded region.

## Precision and execution mode

No autocast or AMP context is used. PyTorch model parameters and transferred model inputs are float32. NumPy Student-t/von-Mises state generation and process-noise construction use float64 intermediates; sampled states are converted to float32 when inserted into the MLP input. The measured student token is float32.

## Hardware and software provenance boundary

The original R2-D runtime archive records assigned CUDA devices (`cuda:0`, `cuda:1`, and `cuda:2`), code/checkpoint hashes, seeds, target counts, and elapsed seconds, but **does not serialize the GPU model, driver, CPU model, RAM, or package lockfile**. Those fields cannot be reconstructed as frozen facts and must not be filled from a later machine snapshot.

For transparency only, the retained `bev` environment at export time reports Python 3.8.20, PyTorch 2.4.1, CUDA build 12.4, cuDNN 90100, CPU `Intel(R) Xeon(R) Platinum 8480+`, approximately 2015.5 GiB RAM, and GPU `unavailable in the export session`. These current values are not asserted to be the original timing host. Consequently, `38.55x` is reproducible from the frozen measurements and code scope, but an exact hardware-normalized latency reproduction requires rerunning the unchanged benchmark with hardware metadata capture.
