# Perception-to-forecast final confirmation

This package executes the frozen final-confirmation protocol for the third
work. It performs no training and cannot authorize new method development.

The permanent order is F0 freeze, F1 descriptive dev analysis, F2 external
transfer, F3 final claim freeze, and F4 one-time official-validation
confirmation. Internal test remains sealed.

## Frozen external conditions

- camera extrinsic rotation S2/S3;
- camera extrinsic translation S3;
- camera temporal offset S2/S3.

The external detector export uses the existing 70-scene dev authority. The
temporal conditions use hash-frozen metadata that mechanically excludes rows
without sufficient camera history. Previously retained external shards based
on official-validation metadata are explicitly ineligible for F2.

## Execution

```bash
cd /home/zhousc/multimodal-uq/BEVFusion
source /home/zhousc/miniconda3/bin/activate bev

python tools/riskprop3d_confirmation_export.py --self-check
python tools/riskprop3d_final_confirmation.py self-check
python tools/riskprop3d_final_confirmation.py freeze
python tools/riskprop3d_final_confirmation.py attributes

# For each frozen external condition and shard 0--3:
python tools/riskprop3d_confirmation_export.py \
  --phase external --condition rotation_s2 --shard-index 0 \
  --shard-count 4 --device 0

python tools/riskprop3d_final_confirmation.py \
  prepare-tracks --phase external --condition rotation_s2

# Replicates may run concurrently on GPUs 0--2.
python tools/riskprop3d_final_confirmation.py \
  external-evaluate --replicate 0 --device 0
python tools/riskprop3d_final_confirmation.py external-finalize
python tools/riskprop3d_final_confirmation.py freeze-claims

# Only after F3, export and build the clean official authority once.
python tools/riskprop3d_confirmation_export.py \
  --phase official --condition clean --shard-index 0 \
  --shard-count 4 --device 0
python tools/riskprop3d_final_confirmation.py \
  prepare-tracks --phase official --condition clean
python tools/riskprop3d_final_confirmation.py \
  official-evaluate --replicate 0 --device 0
python tools/riskprop3d_final_confirmation.py official-finalize
```

All four shards and all three evaluation replicates are mandatory. Retained
outputs are atomic and never overwritten. The access log records the first
official metadata and metric access. No command reads internal test.
