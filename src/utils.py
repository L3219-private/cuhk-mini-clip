def pick_device(device_cfg: str = "auto"):
    """
    if cfg is 'auto': try XPU -> CUDA -> CPU
    otherwise, follow the cfg
    """
    import torch

    if device_cfg != "auto":
        return devise_cfg

    # XPU
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    
    # CUDA
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    # CPU
    return "cpu"
