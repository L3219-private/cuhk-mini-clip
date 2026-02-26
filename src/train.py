from utils import pick_device

# after cfg = load_cfg(...)
# use XPU / CUDA / CPU
device = pick_device(cfg["train"].get("device", "auto"))
print(f"[Info] Using device: {device}")