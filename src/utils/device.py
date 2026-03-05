# src/utils/device.py
def pick_device(prefer = ("xpu", "cuda", "cpu"), device_cfg: str = "auto"):
    """
    device_cfg:
        - "auto": follow prefered order
        - "xpu" / "cuda" / "cpu": force to use xpu / cuda / cpu if available, otherwise fallback to cpu
    """
    import torch

    # not auto
    if device_cfg != "auto":
        if device_cfg == "xpu":
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    return "xpu"
            except Exception:
                pass
        if device_cfg == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device_cfg == "cpu":
            return "cpu"
        return "cpu"

    # auto
    for dev in prefer:
        if dev == "xpu":
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    return "xpu"
            except Exception:
                pass
        elif dev == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif dev == "cpu":
            return "cpu"

    return "cpu"

