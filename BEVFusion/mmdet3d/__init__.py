import torch

if int(torch.__version__.split(".")[0]) >= 2:
    from mmcv.parallel import _functions as mmcv_parallel_functions
    from torch.nn.parallel._functions import _get_stream as torch_get_stream

    # PyTorch 2 expects torch.device while MMCV 1.4 passes an integer GPU id.
    mmcv_parallel_functions._get_stream = lambda device: torch_get_stream(
        torch.device("cuda", device) if isinstance(device, int) else device
    )
