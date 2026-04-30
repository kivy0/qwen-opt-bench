import json
import logging
import time
from pathlib import Path
from typing import Any

import GPUtil
import torch


def setup_logger(experiment_path: Path):
    """
    Project global logger.
    """
    log_file_path = experiment_path / "experiment.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class MetricsLogger:
    def __init__(self, experiment_path: Path) -> None:
        self.filepath = experiment_path / "metrics.jsonl"

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        data = metrics.copy()
        if step is not None:
            data["step"] = step

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")


class HardwareMonitor:
    """
    Collect GPU/CPU metrics on train.
    """

    def __init__(self, device: str) -> None:
        self.torch_device = torch.device(device)
        self.device_str = device
        self.is_cuda = self.torch_device.type == "cuda"
        self._step_start_time: float = 0.0

    def reset(self) -> None:
        """Call at the beginning of each step."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.torch_device)
            torch.cuda.synchronize(self.torch_device)

        self._step_start_time = time.perf_counter()

    def collect(self) -> dict[str, float]:
        """
        Call at the end of the step after all operations.
        Returns a dictionary with hardware metrics.
        """
        if self.is_cuda:
            torch.cuda.synchronize(self.torch_device)

        step_time = time.perf_counter() - self._step_start_time
        metrics: dict[str, float] = {"step_time_sec": round(step_time, 4)}

        if self.is_cuda:
            metrics.update(self._collect_cuda_metrics())

        if self.is_cuda:
            metrics.update(self._collect_gputil_metrics())
        return metrics

    def _collect_cuda_metrics(self) -> dict[str, float]:
        """Memory metrics via torch.cuda."""
        device_idx = (
            self.torch_device.index if self.torch_device.index is not None else 0
        )

        peak_memory_mb = torch.cuda.max_memory_allocated(self.torch_device) / 1024**2
        reserved_memory_mb = torch.cuda.memory_reserved(self.torch_device) / 1024**2
        total_memory_mb = (
            torch.cuda.get_device_properties(device_idx).total_memory / 1024**2
        )

        return {
            "gpu_peak_memory_mb": round(peak_memory_mb, 2),
            "gpu_reserved_memory_mb": round(reserved_memory_mb, 2),
            "gpu_total_memory_mb": round(total_memory_mb, 2),
            "gpu_memory_utilization_pct": round(
                peak_memory_mb / total_memory_mb * 100, 2
            ),
        }

    def _collect_gputil_metrics(self) -> dict[str, float]:
        """GPU utilization via GPUtil."""
        device_idx = (
            self.torch_device.index if self.torch_device.index is not None else 0
        )

        gpus = GPUtil.getGPUs()
        if not gpus or device_idx >= len(gpus):
            return {}

        gpu = gpus[device_idx]
        return {
            "gpu_utilization_pct": round(gpu.load * 100, 2),
        }
