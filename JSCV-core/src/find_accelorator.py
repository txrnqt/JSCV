import platform
from typing import Any, Dict, List

import torch


def find_accelerator() -> Dict[str, Any]:
    """
    Detect available hardware accelerators and GPU devices.

    This function queries the system for supported compute backends such as
    CUDA, cuDNN, and MPS, and collects metadata about detected GPUs.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "gpus" (List[Dict[str, Any]]): Detected GPU devices with
              index and name.
            - "accelerators" (List[str]): Names of available accelerator
              backends (e.g., "CUDA", "cuDNN", "MPS").
            - "os" (str): Host operating system name.
    """
    result = {
        "gpus": [],
        "accelerators": [],
        "os": platform.system(),
    }

    if torch.cuda.is_available():
        result["accelerators"].append("CUDA")
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            result["gpus"].append({"index": i, "name": gpu_name})

    has_cudnn = torch.backends.cudnn.is_available()
    if has_cudnn:
        result["accelerators"].append("cuDNN")

    if torch.backends.mps.is_available():
        result["accelerators"].append("MPS")

    return result
