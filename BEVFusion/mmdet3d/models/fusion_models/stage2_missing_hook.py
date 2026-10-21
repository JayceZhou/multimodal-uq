import torch


class Stage2MissingHook:
    """Zero one sensor BEV tensor immediately before the fuser."""

    def __init__(self, sensor_names, missing_modality):
        if missing_modality not in sensor_names:
            raise ValueError(f"model has no {missing_modality} branch")
        self.index = sensor_names.index(missing_modality)
        self.sensor_names = sensor_names
        self.missing_modality = missing_modality
        self.last_stats = None

    def __call__(self, module, inputs):
        features = list(inputs[0])
        original = features[self.index]
        features[self.index] = torch.zeros_like(original)
        self.last_stats = {
            "missing_modality": self.missing_modality,
            "original_abs_mean": float(original.detach().abs().mean().cpu()),
            "zeroed_abs_max": 0.0,
            "input_abs_mean": {
                name: float(value.detach().abs().mean().cpu())
                for name, value in zip(self.sensor_names, inputs[0])
            },
        }
        return (features,)


def install_stage2_missing_hook(model, corruption_type):
    if model.fuser is None:
        raise ValueError("feature-branch missing requires a fusion model")
    missing = corruption_type.split("_", 1)[0]
    hook = Stage2MissingHook(list(model.encoders.keys()), missing)
    handle = model.fuser.register_forward_pre_hook(hook)

    def capture_fused(module, inputs, output):
        model.stage2_missing["fused_abs_mean"] = float(
            output.detach().abs().mean().cpu()
        )

    fused_handle = model.fuser.register_forward_hook(capture_fused)
    conv = model.fuser[0]
    model.stage2_missing = {
        "hook": hook,
        "handle": handle,
        "fused_handle": fused_handle,
        "modality_present_mask": {
            "camera": missing != "camera",
            "lidar": missing != "lidar",
        },
        "fuser_conv_has_bias": conv.bias is not None,
        "fuser_batchnorm_has_affine_bias": model.fuser[1].bias is not None,
    }
    return model.stage2_missing
