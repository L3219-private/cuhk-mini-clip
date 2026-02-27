# src/
def pick_device(prefer = ("xpu", "cuda", "cpu"), device_cfg: str = "auto"):
    """
    device_cfg:
        - "auto": follow prefered order
        - "xpu" / "cuda" / "cpu": force to use xpu / cuda / cpu if available, otherwise fallback to cpu
    """
    import torch

    if device_cfg != "auto":
        return device_cfg

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
