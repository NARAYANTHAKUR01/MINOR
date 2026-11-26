import torch


def get_device(requested_device: str = "auto") -> str:
    """
    Select the optimal execution device.
    Defaults to Apple Silicon MPS if available, otherwise falls back to CPU.
    """
    req = requested_device.lower().strip()
    if req in ["cpu", "mps"]:
        device = req
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device_name = (
        "Apple Silicon GPU (Metal Performance Shaders)"
        if device == "mps"
        else "Standard CPU"
    )
    print(f"[INFO] Selected compute device: {device.upper()} - {device_name}")
    return device


def get_device_info() -> dict:
    """Return hardware acceleration and PyTorch details."""
    mps_available = bool(torch.backends.mps.is_available())
    mps_built = bool(torch.backends.mps.is_built())
    device = "mps" if mps_available else "cpu"
    device_name = (
        "Apple Silicon GPU (Metal Performance Shaders)"
        if device == "mps"
        else "Standard CPU"
    )

    return {
        "device": device,
        "device_name": device_name,
        "mps_available": mps_available,
        "mps_built": mps_built,
        "torch_version": torch.__version__,
    }
