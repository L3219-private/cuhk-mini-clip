# tests/utils/test_device.py
import torch
import pytest
from src.utils.device import pick_device

# device_cfg = "cpu"
def test_force_cpu():
    assert pick_device(device_cfg="cpu") == "cpu"

# device_cfg = "cuda"
def test_force_cuda(monkeypatch):
    monkeypatch.setattr(torch, "cuda", type("MockCuda", (), {"is_available": lambda: True}))
    assert pick_device(device_cfg="cuda") == "cuda"

# device_cfg = "xpu"
def test_force_xpu(monkeypatch):
    monkeypatch.setattr(torch, "xpu", type("Xpu", (), {"is_available": lambda: True}))
    assert pick_device(device_cfg="xpu") == "xpu"

# device_cfg = "auto"
def test_auto_prefer_cpu():
    # object of class MonkeyPatch
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(torch, "cuda", type("MockCuda", (), {"is_available": lambda: False}))
    monkeypatch.setattr(torch, "xpu", type("MockXpu", (), {"is_available": lambda: False}))
    assert pick_device(prefer=("cpu", "cuda")) == "cpu"  # cpu fallback
    monkeypatch.undo()